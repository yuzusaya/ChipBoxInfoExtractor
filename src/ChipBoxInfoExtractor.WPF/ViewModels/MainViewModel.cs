using Microsoft.Win32;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using ChipBoxInfoExtractor.ConsoleApp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using CommunityToolkit.Mvvm.Input;

namespace ChipBoxInfoExtractor.WPF.ViewModels;

public enum BrowseTypes
{
    File,
    Folder,
}
public enum FileTypes
{
    Image,
    Video,
    Pdf,
    Excel,
    Audio,
}

public partial class MainViewModel : BaseViewModel
{
  

    [ObservableProperty]
    private bool _isProcessing;
    [ObservableProperty]
    private string _filePathBeingProcess;
    [ObservableProperty]
    private string _message;
    [RelayCommand(AllowConcurrentExecutions =false)]

    private async Task BrowseFileOrFolder(BrowseTypes browseType)
    {
        if (IsProcessing)
            return;
        IsProcessing = true;
        if (browseType == BrowseTypes.File)
        {
            var filePath = BrowseFile();
            if (!string.IsNullOrEmpty(filePath))
            {
                FilePathBeingProcess = filePath;
                await ExtractBoxesAndSave(filePath, AppendProcessedSuffixToFileName(filePath));
            }
        }
        else if (browseType == BrowseTypes.Folder)
        {
            var folderPath = await BrowseFolder("");
            if (!string.IsNullOrEmpty(folderPath))
            {
                await LoopFolderRecursively(folderPath);
            }
        }
        IsProcessing = false;
    }
    private string BrowseFile()
    {
        OpenFileDialog op = new OpenFileDialog();
        //todo filter based on file types
        op.Title = "Select a picture";
        op.Filter = "All supported graphics|*.jpg;*.jpeg;*.png|" +
          "JPEG (*.jpg;*.jpeg)|*.jpg;*.jpeg|" +
          "Portable Network Graphic (*.png)|*.png";
        if (op.ShowDialog() is true)
        {
            return op.FileName;
        }

        return string.Empty;
    }
    private async Task<string> BrowseFolder(string initialDirectoryPath)
    {
        OpenFolderDialog dialog = new OpenFolderDialog();
        dialog.Title = "Select a folder";
        dialog.InitialDirectory = initialDirectoryPath;
        if (dialog.ShowDialog() is true)
        {
            return dialog.FolderName;
        }
        return string.Empty;
    }

    //extract to static method
    public string AppendProcessedSuffixToFileName(string fileName)
    {
        return fileName.Insert(fileName.LastIndexOf('.'), "_processed");
    }
    //extract to static method
    public string AddFileToProcessedFolder(string fileName)
    {
        return fileName.Insert(fileName.LastIndexOf('\\'), @"\\processed");
    }

    private async Task LoopFolderRecursively(string path, List<string> suffixesBeingIgnored = null)
    {
        var directories = Directory.GetDirectories(path);
        foreach (string directory in directories)
        {
            if (directory.EndsWith("processed") || directory.EndsWith("failed"))
            {
                continue;
            }
            await LoopFolderRecursively(directory, suffixesBeingIgnored);
        }
        foreach (string fileName in Directory.GetFiles(path))
        {
            if (suffixesBeingIgnored?.Any(suffix => fileName.EndsWith(suffix, StringComparison.InvariantCultureIgnoreCase)) == true)
                continue;
            FilePathBeingProcess = fileName;

            var processedFolderPath = Directory.CreateDirectory(path + @"\\processed").Name;
            //this is the file that you want to process and store, for example generate a thumbnail from image
            string newFileName = AddFileToProcessedFolder(fileName);
            newFileName = AppendProcessedSuffixToFileName(newFileName);
            //do some processing and use newFileName to store the processed files

            //this is the file that you want to record the processed data, for example store the results in excel
            //string resultTxtPath = processedFolderPath + @"\result.txt";
            //using (StreamWriter sw = File.AppendText(resultTxtPath))
            //{
            //    sw.WriteLine(Path.GetFileName(fileName));
            //}
            try
            {
                await ExtractBoxesAndSave(fileName, newFileName);
            }
            catch (Exception ex)
            {
                string exceptionTxtPath = processedFolderPath + @"\exception.txt";
                using (StreamWriter sw = File.AppendText(exceptionTxtPath))
                {
                    sw.WriteLine(Path.GetFileName(fileName) + "\t" + ex.Message);
                }
            }

        }
        try
        {
            string overAllResultTxtPath = path + @"\processed\result.txt";
            using (StreamWriter sw = File.AppendText(overAllResultTxtPath))
            {

            }
        }
        catch (Exception e)
        {

        }
        finally
        {

        }
    }
    private readonly ObjectDetection _objectDetection;
    public MainViewModel()
    {
        _objectDetection = new ObjectDetection(new List<string>()
        {
            "box"
        });
        _objectDetection.Init("model.onnx");
    }

    private async Task ExtractBoxesAndSave(string filePath, string processedFilePath)
    {
        var boundingBoxes = await _objectDetection.PredictImageAsync(filePath);
        boundingBoxes = boundingBoxes.Where(x => x.Probability > 0.5).ToList();
        for (int i = 0; i < boundingBoxes.Count; i++)
        {
            var image = SegmentImage(filePath, boundingBoxes[i].BoundingBox);
            image.Save(processedFilePath.Replace("_processed", $"_processed{i+1}"));
        }
    }

    public static Image<Rgba32> SegmentImage(string imagePath, BoundingBox boundingBox)
    {
        // Load the image from file
        using var image = Image.Load<Rgba32>(imagePath);

        // Get image dimensions
        int imageWidth = image.Width;
        int imageHeight = image.Height;

        // Convert bounding box normalized coordinates to pixel values
        int left = (int)(boundingBox.Left * imageWidth);
        int top = (int)(boundingBox.Top * imageHeight);
        int width = (int)(boundingBox.Width * imageWidth);
        int height = (int)(boundingBox.Height * imageHeight);

        // Ensure the dimensions are within the bounds of the image
        left = Math.Max(0, left);
        top = Math.Max(0, top);
        width = Math.Min(width, imageWidth - left);
        height = Math.Min(height, imageHeight - top);

        // Create the cropped image using the bounding box
        var cropRectangle = new Rectangle(left, top, width, height);
        //var croppedImage = image.Clone(ctx => ctx.Crop(cropRectangle));
        var clonedImage = image.Clone();
        clonedImage.Mutate(i => i.Crop(cropRectangle));
        return clonedImage;
    }
}
