from pathlib import Path
import subprocess
import sys
import re
import pkg_resources
from huggingface_hub import hf_hub_download


class SAMModelManager:
    """Gestor unificado para SAM, MobileSAM, HQ-SAM/Light-HQ-SAM y MedSAM(+2)."""

    _MODELS = {
        # 1) Segment Anything original ---------------------------------------
        "sam": {
            "pip": "git+https://github.com/facebookresearch/segment-anything.git",
            "variants": {
                "vit_b": ("url", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
                "vit_l": ("url", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
                "vit_h": ("url", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
            },
        },
        # 2) MobileSAM --------------------------------------------------------
        "mobilesam": {
            "pip": "git+https://github.com/ChaoningZhang/MobileSAM.git",
            "variants": {
                "vit_t": ("dhkim2810/MobileSAM", "mobile_sam.pt"),  # 40 MB
            },
        },
        # 3) HQ-SAM / Light-HQ-SAM -------------------------------------------
        "hq_sam": {
            "pip": "git+https://github.com/SysCV/sam-hq.git",
            "variants": {
                "vit_l": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"),
                "vit_t": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth"),
                "vit_h": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"),
                "vit_b": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"),
                "hiera_l_2.1": ("url", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam2.1_hq_hiera_large.pt"),
            },
        },
        # 4) MedSAM y MedSAM2 -------------------------------------------------
        "medsam": {
            "pip": "git+https://github.com/bowang-lab/MedSAM.git",
            "variants": {
                "medsam1": ("url", "https://zenodo.org/records/10689643/files/medsam_vit_b.pth"),
                "medsam2_latest": ("url", "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"),
            },
        },
    }

    def __init__(self, root_dir: str):
        self.root = Path(root_dir).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------- API pública ------------------------------
    def install_repo(self, family: str) -> None:
        """Install the repository for a model family."""
        if family not in self._MODELS:
            raise ValueError(f"Unknown model family: {family}. Available: {list(self._MODELS.keys())}")
        
        pkg = self._MODELS[family]["pip"]
        if not self._pkg_installed(pkg):
            print(f"Instalando {family}…")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install {family}: {e}")
        else:
            print(f"{family} ya estaba instalado.")

    def download_variant(self, family: str, variant: str, force: bool = False) -> Path:
        """Download a specific variant of a model family."""
        if family not in self._MODELS:
            raise ValueError(f"Unknown model family: {family}")
        
        if variant not in self._MODELS[family]["variants"]:
            available = list(self._MODELS[family]["variants"].keys())
            raise ValueError(f"Unknown variant {variant} for {family}. Available: {available}")
        
        repo_or_url, filename = self._MODELS[family]["variants"][variant]
        dest = self.root / Path(filename).name

        if dest.exists() and not force:
            print(f"{dest.name} ya existe → se omite descarga.")
            return dest

        try:
            if repo_or_url == "url" or repo_or_url.startswith("http"):
                import requests
                
                url = filename if repo_or_url == "url" else repo_or_url
                print(f"Descargando desde URL directa:\n  {url}")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                print(f"Descargando desde Hugging Face ({repo_or_url})…")
                dest_path = hf_hub_download(
                    repo_id=repo_or_url,
                    filename=filename,
                    cache_dir=str(self.root),
                    force_download=force,
                )
                dest = Path(dest_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {family}/{variant}: {e}")

        return dest

    def setup(self, family: str, variant: str, install: bool = True, force: bool = False) -> Path:
        """Complete setup: install repository and download model variant."""
        if install:
            self.install_repo(family)
        return self.download_variant(family, variant, force)

    def list_supported(self) -> None:
        """Print all supported model families and their variants."""
        print("Supported models:")
        for fam, info in self._MODELS.items():
            variants = ', '.join(info['variants'].keys())
            print(f"  {fam}: {variants}")

    @staticmethod
    def _pkg_installed(pip_spec: str) -> bool:
        """Check if a pip package is installed."""
        name = re.sub(r".*/|\.git$", "", pip_spec)
        try:
            pkg_resources.get_distribution(name)
            return True
        except pkg_resources.DistributionNotFound:
            return False


if __name__ == "__main__":  # pragma: no cover - ejemplo de uso
    mgr = SAMModelManager("Models")
    mgr.list_supported()
