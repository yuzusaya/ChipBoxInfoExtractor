using Microsoft.Win32;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;

namespace ChipBoxInfoExtractor.WPF.ViewModels;

public partial class MainViewModel : BaseViewModel
{
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

    [ObservableProperty] 
    private bool _isProcessing;
    [ObservableProperty]
    private string _filePathBeingProcess;
    [ObservableProperty]
    private string _message;

    private void BrowseFileOrFolder(BrowseTypes browseType)
    {
        if (IsProcessing)
            return;
        IsProcessing = true;
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

    private async Task LoopFolderRecursively(string path, List<string> suffixesBeingIgnored)
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
            if (suffixesBeingIgnored.Any(suffix => fileName.EndsWith(suffix, StringComparison.InvariantCultureIgnoreCase)))
                continue;
            FilePathBeingProcess = fileName;

            var processedFolderPath = Directory.CreateDirectory(path + @"\\processed").Name;
            //this is the file that you want to process and store, for example generate a thumbnail from image
            string newFileName = AddFileToProcessedFolder(fileName);
            newFileName = AppendProcessedSuffixToFileName(newFileName);
            //do some processing and use newFileName to store the processed files

            //this is the file that you want to record the processed data, for example store the results in excel
            string resultTxtPath = processedFolderPath + @"\result.txt";
            using (StreamWriter sw = File.AppendText(resultTxtPath))
            {
                sw.WriteLine(Path.GetFileName(fileName));
            }
            try
            {

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

}
