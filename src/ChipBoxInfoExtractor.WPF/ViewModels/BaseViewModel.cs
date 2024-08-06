using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace ChipBoxInfoExtractor.WPF.ViewModels;
public partial class BaseViewModel : ObservableObject, INotifyPropertyChanged
{

    [ObservableProperty]
    string _title;

    [ObservableProperty]
    bool _isBusy;

    [ObservableProperty]
    string _currentState = StateKey.None;

    [ObservableProperty]
    bool _canStateChange;

    protected bool SetProperty<T>(ref T backingStore, T value, [CallerMemberName] string propertyName = "", Action onChanged = null)
    {
        if (EqualityComparer<T>.Default.Equals(backingStore, value))
            return false;
        backingStore = value;
        onChanged?.Invoke();
        OnPropertyChanged(propertyName);
        return true;
    }
}

public class StateKey
{
    public const string Loading = "Loading";
    public const string Success = "Success";
    public const string Error = "Error";
    public const string None = "None";
}

