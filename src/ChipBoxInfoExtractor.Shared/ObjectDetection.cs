using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

// The steps implemented in the object detection sample code: 
// 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
// 2. resize network input size to (w', h')
// 3. pass the image to network and do inference
// (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
// <copyright file="ObjectDetection.cs" company="Microsoft Corporation">
// Copyright (c) Microsoft Corporation. All rights reserved.
// </copyright>

/// Script for CustomVision's exported object detection model.

namespace ChipBoxInfoExtractor.ConsoleApp;

public sealed class BoundingBox
{
    public BoundingBox(float left, float top, float width, float height)
    {
        this.Left = left;
        this.Top = top;
        this.Width = width;
        this.Height = height;
    }

    public float Left { get; private set; }
    public float Top { get; private set; }
    public float Width { get; private set; }
    public float Height { get; private set; }
}

public sealed class PredictionModel
{
    public PredictionModel(float probability, string tagName, BoundingBox boundingBox)
    {
        this.Probability = probability;
        this.TagName = tagName;
        this.BoundingBox = boundingBox;
    }

    public float Probability { get; private set; }
    public string TagName { get; private set; }
    public BoundingBox BoundingBox { get; private set; }
}

public class ObjectDetection
{
    private static readonly float[] Anchors = new float[] { 0.573f, 0.677f, 1.87f, 2.06f, 3.34f, 5.47f, 7.88f, 3.53f, 9.77f, 9.17f };
    private readonly IList<string> labels;
    private readonly int maxDetections;
    private readonly float probabilityThreshold;
    private readonly float iouThreshold;
    private InferenceSession session;
    private const int imageInputSize = 512 * 512;
    //private const int imageInputSize = 416 * 416;

    public ObjectDetection(IList<string> labels, int maxDetections = 20, float probabilityThreshold = 0.1f, float iouThreshold = 0.45f)
    {
        this.labels = labels;
        this.maxDetections = maxDetections;
        this.probabilityThreshold = probabilityThreshold;
        this.iouThreshold = iouThreshold;
    }

    /// <summary>
    /// Initialize
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file</param>
    public void Init(string modelPath)
    {
        this.session = new InferenceSession(modelPath);

        Debug.Assert(this.session.InputMetadata.Count == 1, "The number of input must be 1");
        Debug.Assert(this.session.OutputMetadata.Count == 1, "The number of output must be 1");
    }

    /// <summary>
    /// Detect objects from the given image.
    /// </summary>
    public async Task<IList<PredictionModel>> PredictImageAsync(string imagePath)
    {
        //using (var image = Image.Load<Rgba32>(imagePath))
        //{
        //    int imageWidth = image.Width;
        //    int imageHeight = image.Height;

        //    double ratio = Math.Sqrt((double)imageInputSize / (double)imageWidth / (double)imageHeight);
        //    int targetWidth = 32 * (int)Math.Round(imageWidth * ratio / 32);
        //    int targetHeight = 32 * (int)Math.Round(imageHeight * ratio / 32);
        //    image.Mutate(ctx => ctx.Resize(targetWidth, targetHeight));

        //    var tensor = ImageToTensor(image);

        //    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", tensor) };

        //    using (var results = session.Run(inputs))
        //    {
        //        var output = results.First().AsTensor<float>();
        //        return Postprocess(output);
        //    }
        //}

        var tensor = LoadInputTensor(new FileInfo(imagePath), 416);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", tensor) };

        using (var results = session.Run(inputs))
        {
            var output = results.First().AsTensor<float>();
            return Postprocess(output);
        }
    }

    // Load an image file and create a RGB[0-255] tensor.
    private Tensor<float> LoadInputTensor(FileInfo imageFilepath, int imageSize)
    {
        var input = new DenseTensor<float>(new[] { 1, 3, imageSize, imageSize });
        using (var image = Image.Load<Rgb24>(imageFilepath.ToString()))
        {
            image.Mutate(x => x.Resize(imageSize, imageSize));

            image.ProcessPixelRows(pixelAccessor =>
            {
                for (int y = 0; y < pixelAccessor.Height; y++)
                {
                    Span<Rgb24> row = pixelAccessor.GetRowSpan(y);

                    // Using row.Length helps JIT to eliminate bounds checks when accessing row[x].
                    for (int x = 0; x < row.Length; x++)
                    {
                        input[0, 0, y, x] = row[x].R;
                        input[0, 1, y, x] = row[x].G;
                        input[0, 2, y, x] = row[x].B;
                    }
                }
            });
        }
        return input;
    }

    private Tensor<float> ImageToTensor(Image<Rgba32> image)
    {
        var width = image.Width;
        var height = image.Height;
        var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var pixel = image[x, y];
                tensor[0, 0, y, x] = pixel.R / 255.0f;
                tensor[0, 1, y, x] = pixel.G / 255.0f;
                tensor[0, 2, y, x] = pixel.B / 255.0f;
            }
        }

        return tensor;
    }

    private List<PredictionModel> Postprocess(Tensor<float> predictionOutput)
    {
        var extractedBoxes = ExtractBoxes(predictionOutput, ObjectDetection.Anchors);
        return SuppressNonMaximum(extractedBoxes);
    }

    private ExtractedBoxes ExtractBoxes(Tensor<float> predictionOutput, float[] anchors)
    {
        var shape = predictionOutput.Dimensions;
        Debug.Assert(shape.Length == 4, "The model output has unexpected shape");
        Debug.Assert(shape[0] == 1, "The batch size must be 1");

        var outputs = predictionOutput.ToArray();

        var numAnchor = anchors.Length / 2;
        var channels = shape[1];
        var height = shape[2];
        var width = shape[3];

        Debug.Assert(channels % numAnchor == 0);
        var numClass = (channels / numAnchor) - 5;

        Debug.Assert(numClass == this.labels.Count);

        var boxes = new List<BoundingBox>();
        var probs = new List<float[]>();
        for (int gridY = 0; gridY < height; gridY++)
        {
            for (int gridX = 0; gridX < width; gridX++)
            {
                int offset = 0;
                int stride = height * width;
                int baseOffset = gridX + gridY * width;

                for (int i = 0; i < numAnchor; i++)
                {
                    var x = (Logistic(outputs[baseOffset + (offset++ * stride)]) + gridX) / width;
                    var y = (Logistic(outputs[baseOffset + (offset++ * stride)]) + gridY) / height;
                    var w = (float)Math.Exp(outputs[baseOffset + (offset++ * stride)]) * anchors[i * 2] / width;
                    var h = (float)Math.Exp(outputs[baseOffset + (offset++ * stride)]) * anchors[i * 2 + 1] / height;

                    x = x - (w / 2);
                    y = y - (h / 2);

                    var objectness = Logistic(outputs[baseOffset + (offset++ * stride)]);

                    var classProbabilities = new float[numClass];
                    for (int j = 0; j < numClass; j++)
                    {
                        classProbabilities[j] = outputs[baseOffset + (offset++ * stride)];
                    }
                    var max = classProbabilities.Max();
                    for (int j = 0; j < numClass; j++)
                    {
                        classProbabilities[j] = (float)Math.Exp(classProbabilities[j] - max);
                    }
                    var sum = classProbabilities.Sum();
                    for (int j = 0; j < numClass; j++)
                    {
                        classProbabilities[j] *= objectness / sum;
                    }

                    if (classProbabilities.Max() > this.probabilityThreshold)
                    {
                        boxes.Add(new BoundingBox(x, y, w, h));
                        probs.Add(classProbabilities);
                    }
                }
                Debug.Assert(offset == channels);
            }
        }

        Debug.Assert(boxes.Count == probs.Count);
        return new ExtractedBoxes(boxes, probs);
    }

    private List<PredictionModel> SuppressNonMaximum(ExtractedBoxes extractedBoxes)
    {
        var predictions = new List<PredictionModel>();

        if (extractedBoxes.Probabilities.Count > 0)
        {
            var maxProbs = extractedBoxes.Probabilities.Select(x => x.Max()).ToArray();

            while (predictions.Count < this.maxDetections)
            {
                var max = maxProbs.Max();
                if (max < this.probabilityThreshold)
                {
                    break;
                }
                var index = Array.IndexOf(maxProbs, max);
                var maxClass = Array.IndexOf(extractedBoxes.Probabilities[index], max);

                predictions.Add(new PredictionModel(max, this.labels[maxClass], extractedBoxes.Boxes[index]));

                for (int i = 0; i < extractedBoxes.Boxes.Count; i++)
                {
                    if (CalculateIOU(extractedBoxes.Boxes[index], extractedBoxes.Boxes[i]) > this.iouThreshold)
                    {
                        extractedBoxes.Probabilities[i][maxClass] = 0;
                        maxProbs[i] = extractedBoxes.Probabilities[i].Max();
                    }
                }
            }
        }

        return predictions;
    }

    private static float Logistic(float x)
    {
        if (x > 0)
        {
            return (float)(1 / (1 + Math.Exp(-x)));
        }
        else
        {
            var e = Math.Exp(x);
            return (float)(e / (1 + e));
        }
    }

    private static float CalculateIOU(BoundingBox box0, BoundingBox box1)
    {
        var x1 = Math.Max(box0.Left, box1.Left);
        var y1 = Math.Max(box0.Top, box1.Top);
        var x2 = Math.Min(box0.Left + box0.Width, box1.Left + box1.Width);
        var y2 = Math.Min(box0.Top + box0.Height, box1.Top + box1.Height);
        var w = Math.Max(0, x2 - x1);
        var h = Math.Max(0, y2 - y1);

        return w * h / ((box0.Width * box0.Height) + (box1.Width * box1.Height) - (w * h));
    }

    private class ExtractedBoxes
    {
        public List<BoundingBox> Boxes { get; }
        public List<float[]> Probabilities { get; }

        public ExtractedBoxes(List<BoundingBox> boxes, List<float[]> probs)
        {
            this.Boxes = boxes;
            this.Probabilities = probs;
        }
    }
}

