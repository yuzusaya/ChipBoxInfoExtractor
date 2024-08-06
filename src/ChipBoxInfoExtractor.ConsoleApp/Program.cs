using ChipBoxInfoExtractor.Shared;

ObjectDetection objDetection = new ObjectDetection(new List<string>()
        {
            "box"
        });
objDetection.Init("model.onnx");


var boxes = await objDetection.PredictImageAsync("image.JPG");
boxes = boxes.OrderByDescending(x => x.Probability).Where(x => x.Probability > 0.5).ToList();
foreach (var box in boxes)
{
    Console.WriteLine($"{box.Probability}");
}
