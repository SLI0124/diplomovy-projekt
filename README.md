# Diplomový projekt

*Cílem této práce je představit a popsat metody předpovídání časových řad s využitím hlubokých neuronových sítí.*

V tomto repozitáři najdete ukázky a popisy implementace metod pro předpovídání časových řad s využitím hlubokého učení. Zaměřuji se zde na praktickou stránku práce s kódem, jeho ověřování na reálných datech a srovnání dosažených výsledků. Některé natrénované modely mohou být poměrně velké, a proto nebudou součástí tohoto repozitáře.

## ⚙️ Instalace

### Předpoklady

- **Python 3.8+** by měl postačit
- **CUDA Runtime** (pro GPU podporu) - viz [oficiální instalační průvodce](https://developer.nvidia.com/cuda-downloads)

### 📦 Instalace pomocí pip

```bash
# Instalace všech závislostí
pip install -r requirements.txt
```

> 💡 **Pro jiné verze CUDA:** Navštivte [oficiální stránky PyTorch](https://pytorch.org/get-started/locally/) a nahraďte torch instalaci správnou verzí.

### 🐍 Instalace pomocí Anaconda/Conda

```bash
# Vytvoření a aktivace prostředí
conda env create -f environment.yml
conda activate master-thesis-sli0124
# jelikož conda nemá všechny balíčky, je potřeba doinstalovat balíčky přes pip
pip install -r requirements.txt
```

### ✅ Ověření instalace

```bash
python check_gpu.py
```

> ⚠️ **Řešení problémů s DLL:**
>
> Při instalaci se může objevit [chyba s DLL](https://discuss.pytorch.org/t/importerror-dll-load-failed-while-importing-c-das-angegebene-modul-wurde-nicht-gefunden-the-specified-module-can-not-be-found/217569), která je poměrně častá. Tento problém jsem vyřešil instalací konkrétní stabilní verze, nebo jakékoli předchozí verze PyTorch. Všechny dostupné verze najdete na [stránce s předchozími verzemi PyTorch](https://pytorch.org/get-started/previous-versions/).
>
> ```bash
> pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
> ```

## Použité datasety

Svoboda, R., et al. (2024). *Natural Gas Consumption Forecasting System*.  
Dataset AI VŠB – Natural Gas Forecasting. Dostupné z tohoto [odkazu](https://ai.vsb.cz/natural-gas-forecasting).  
DOI: [10.1016/j.energy.2020.119430](https://doi.org/10.1016/j.energy.2020.119430)
