# Diplomový projekt

*Cílem této práce je představit a popsat metody předpovídání časových řad s využitím hlubokých neuronových sítí.*

V tomto repozitáři najdete ukázky a popisy implementace metod pro předpovídání časových řad s využitím hlubokého učení. Zaměřuji se zde na praktickou stránku práce s kódem, jeho ověřování na reálných datech a srovnání dosažených výsledků. Některé natrénované modely mohou být poměrně velké, a proto nebudou součástí tohoto repozitáře.

Součástí tohoto repozitáře je také sběr dat, na kterých je vše postaveno.

## ⚙️ Instalace

### Předpoklady

- **Python 3.10+**
- **CUDA Runtime** (pro GPU podporu) - viz [oficiální instalační průvodce](https://developer.nvidia.com/cuda-downloads)

### 📦 Instalace pomocí pip

```bash
# Instalace všech závislostí
pip install -r requirements.txt
```

> 💡 **Pro podporu určité verze CUDA:** Navštivte [oficiální stránky PyTorch](https://pytorch.org/get-started/locally/) a nahraďte torch instalaci správnou verzí.

### 🐍 Instalace pomocí Anaconda/Conda

```bash
# Vytvoření a aktivace prostředí
conda env create -f environment.yml
conda activate master-thesis-sli0124
```

### ✅ Ověření instalace

```bash
python tools/check_gpu.py
```

> ⚠️ **Řešení problémů s DLL:**
>
> Při instalaci se může objevit [chyba s DLL](https://discuss.pytorch.org/t/importerror-dll-load-failed-while-importing-c-das-angegebene-modul-wurde-nicht-gefunden-the-specified-module-can-not-be-found/217569), která je poměrně častá. Tento problém jsem vyřešil instalací konkrétní stabilní verze, nebo jakékoli předchozí verze PyTorch. Všechny dostupné verze najdete na [stránce s předchozími verzemi PyTorch](https://pytorch.org/get-started/previous-versions/). Pro CUDA 12.8 jsem použil následující příkaz a verzi PyTorch 2.8.0:
>
> ```bash
> pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
> ```
