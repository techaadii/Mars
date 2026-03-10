import numpy as np
import pandas as pd
import torch
from typing import TypedDict, Literal,Sized
from pathlib import Path
from tifffile import imread as tiff_imread
import json


def _load_config(config_path: Path)->dict:
    with open(config_path,'r') as f:
        return json.load(f)


config_path: Path = Path("/home/moonlab/Mars/src/utils/data/segmunich_config.json")
    
cfg =_load_config(config_path=config_path)





type _SplitName = Literal["train","val","test"]

class SegMunich_Sample(TypedDict):
    """
    Class defines the custom datatype of each of the bands that are there.
    All 10 sentinel-2 bands, shape(10,128,128), dtype=float32
    Bands in order: B01 ,B02 ,B03 ,B04 ,B05 ,B06 ,B07 ,B8A ,B11 ,B12
    """
    optical:torch.Tensor

    rgb:torch.Tensor
    """True colour stack, Shape (3,128,128). Corresponding bands: B04 (Red), B03(Green), B02(Blue)"""

    nir:torch.Tensor
    """Near infrared red, Shape(128,128). Corresponding bands: B8A"""

    swir:torch.Tensor
    """Short wave infrared, Shape(2,128,128). Correspoding bands: B11 ,B12"""

    veg_red_edge:torch.Tensor
    """Vegetation red edge,shape(3,128,128). Corresponding bands: B05 ,B06 ,B07"""

    coastal_aerosol:torch.Tensor
    """Coastal Aerosol,shape(128,128). Corresponding bands: B01"""

    optical_channel_wv:torch.Tensor
    """Central Wavelength for each of the 10 bands in nm, shape(10,)"""

    spatial_resolution:int
    """Ground Sampling distance in meters (always 10 for this dataset)"""

    label:torch.Tensor | None
    """Remapped segmentation masks for each of the following classes,shape(128,128)
       None in case if masks are not available for the test dataset."""
    


class SegMunich_Dataset(Sized,torch.utils.data.Dataset[SegMunich_Sample]):
    """Dataset class for the SegMunich dataset"""

    _B01_COASTAL_AEROSOL_INDEX : int = 0
    _B02_BLUE_INDEX            : int = 1
    _B03_GREEN_INDEX           : int = 2
    _B04_RED_INDEX             : int = 3
    _B05_VRE_INDEX             : int = 4
    _B06_VRE_INDEX             : int = 5
    _B07_VRE_INDEX             : int = 6
    _B8A_NIR_INDEX             : int = 7
    _B11_SWIR_INDEX            : int = 8
    _B12_SWIR_INDEX            : int = 9

    _VEG_RED_EDGE              : slice = slice(4,7)
    _SWIR_SLICE                : slice = slice(8,10)





    def __init__(
            self,
            data_root:Path,
            split:_SplitName,
            config_path: Path,
            apply_remap:bool=True,
            
        )->None:
        
        """
        Parameters
        ----------
        data_root:
            Path to the extracted SegMunich directory that contains
            ``metadata.csv`` (i.e. the ``SegMunich/`` folder inside the zip).
        split:
            One of ``"train"``, ``"val"``, or ``"test"``.
        apply_remap:
            If ``True`` (default), raw label values are remapped according to
            the official mapping defined in ``SegMunich.py``.
            Set to ``False`` to keep original values from the .tif masks.
        """

        super().__init__()
        

        # Dataset Config
        self._inner_dir = cfg["dataset"]["inner_dir"]
        self._metadata_file = cfg["dataset"]["metadata_file"]
        self._num_classes = cfg["dataset"]["num_classes"]
        self._spatial_resolution = cfg["dataset"]["spatial_resolution"]
        self._img_size = cfg["dataset"]["image_size"]

        # Band Config
        self._channel_wv = torch.Tensor(cfg["bands"]["channel_wv"])
        self.S2_MEAN = cfg["bands"]["mean"]
        self.S2_STD = cfg["bands"]["mean"]

        self._label_remap: dict[int, int] = {
            int(k): int(v) for k, v in cfg["label_remap"].items()
        }


        self._data_root : Path = Path(data_root) / self._inner_dir
        self._split : _SplitName = split
        self._apply_remap : bool = apply_remap
        
        self.metadata_path : Path = self._data_root / self._metadata_file

        self.full_meta : pd.DataFrame = pd.read_csv(self.metadata_path)

        self._meta : pd.DataFrame = (
            self.full_meta[self.full_meta["split"] == split]
            .reset_index(drop=True)
        )

        
    def __len__(self)->int:
        return len(self._meta)

    def __getitem__(self, index:int)->SegMunich_Sample:
        row = self._meta.iloc[index]

        optical_path : Path = self._data_root / row["optical_path"]
        optical_array : np.ndarray = tiff_imread(str(optical_path))
        if optical_array.ndim==3:
            optical_array = np.transpose(optical_array, (2, 0, 1))
        optical: torch.Tensor = torch.from_numpy(optical_array.copy()).to(torch.float32)

        label: torch.Tensor | None = None
        label_path_str : str = row.get("label_path","")
        if pd.notna(label_path_str) and label_path_str != "":
            label_path: Path = self._data_root / label_path_str
            if label_path.exists():
                label = torch.from_numpy(tiff_imread(str(label_path)).copy()).to(torch.int32)
                if self._apply_remap:
                    remapped = label.clone()
                    for raw_val, class_idx in self._label_remap.items():
                        remapped[label == raw_val] = class_idx
                    label = remapped



        rgb: torch.Tensor             = optical[[self._B04_RED_INDEX, self._B03_GREEN_INDEX, self._B02_BLUE_INDEX]]
        nir: torch.Tensor             = optical[self._B8A_NIR_INDEX]
        swir: torch.Tensor            = optical[self._SWIR_SLICE]
        veg_red_edge: torch.Tensor    = optical[self._VEG_RED_EDGE]
        coastal_aerosol: torch.Tensor = optical[self._B01_COASTAL_AEROSOL_INDEX]
        
        
        return SegMunich_Sample(
            optical=optical, # Shape (10,128,128)
            rgb=rgb,
            nir=nir,
            swir=swir,
            veg_red_edge=veg_red_edge,
            coastal_aerosol=coastal_aerosol,
            optical_channel_wv=self._channel_wv,
            spatial_resolution=cfg["dataset"]["spatial_resolution"],
            label=label
        )
   