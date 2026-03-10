import numpy as np
import pandas as pd
import torch
from typing import TypedDict, Literal,Sized
from pathlib import Path
from tifffile import imread as tiff_imread


S2_MEAN: list[float] = [
    752.40087073, 884.29673756, 1144.16202635, 1297.47289228,
    1624.90992062, 2194.6423161, 2422.21248945, 2581.64687018,
    2368.51236873, 1805.06846033,
]
S2_STD: list[float] = [
    1108.02887453, 1155.15170768, 1183.6292542,  1368.11351514,
    1370.265037,   1355.55390699, 1416.51487101, 1439.3086061,
    1455.52084939, 1343.48379601,
]

S2_CHANNEL_WV: list[float] = [
    442.7, 492.4, 559.8, 664.6, 704.1,
    740.5, 782.8, 864.7, 1613.7, 2202.4,
]

_LABEL_REMAP: dict[int, int] = {
    21: 1,
    22: 2,
    23: 3,
    31: 4,
    32: 6,
    33: 7,
    41: 8,
    13: 9,
    14: 10,
}


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

    _METADATA_FILE : str = "Add the metadata.csv path"
    _INNER_DIR     : str = " Add the segmunich directory path"

    NUM_CLASSES    : int = 13
    SPATIAL_RESOLUTION : int = 10 #(in meters)
    IMG_SIZE : int =128


    def __init__(
            self,
            data_root:Path,
            split:_SplitName,
            apply_remap:bool=True
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
        self._data_root : Path = Path(data_root) / self._INNER_DIR
        self._split : _SplitName = split
        self._apply_remap : bool = apply_remap
        
        self.metadata_path : Path = Path(data_root) / self._METADATA_FILE

        self.full_meta : pd.DataFrame = pd.read_csv(self.metadata_path)

        self._meta : pd.DataFrame = (
            self.full_meta[self.full_meta["split"] == split]
            .reset_index(drop=True)
        )

        self._channel_wv: torch.Tensor = torch.tensor(
            S2_CHANNEL_WV, dtype=torch.float32
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
                    for raw_val, class_idx in _LABEL_REMAP.items():
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
            spatial_resolution=self.SPATIAL_RESOLUTION,
            label=label
        )
        




























































       