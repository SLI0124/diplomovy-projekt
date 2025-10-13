# Diplomový projekt

*Cílem této práce je představit a popsat metody předpovídání časových řad s využitím hlubokých neuronových sítí.*

V tomto repozitáři najdete ukázky a popisy implementace metod pro předpovídání časových řad s využitím hlubokého učení. Zaměřuji se zde na praktickou stránku práce s kódem, jeho ověřování na reálných datech a srovnání dosažených výsledků. Některé natrénované modely mohou být poměrně velké, a proto nebudou součástí tohoto repozitáře.

## ⚙️ Instalace

Pro implementaci projektu byla zvolena knihovna **PyTorch**, která nabízí výbornou podporu pro práci s grafickými akcelerátory. Instalace a konfigurace této knihovny může být někdy složitější, proto **doporučuji řídit se oficiální dokumentací** dostupnou [zde](https://pytorch.org/get-started/locally/).

### 📦 Standardní instalace

PyTorch lze nainstalovat standardním způsobem pomocí **pip**. Tento způsob je nejjednodušší a oficiálně podporovaný, ale vyžaduje instalaci ovladačů pro grafickou kartu a CUDA. Stačí se řídit oficiální dokumentací zmíněnou výše.

### 🐍 Anaconda

PyTorch byl dlouhou dobu dostupný k instalaci přes **Anacondu**, avšak tento způsob instalace byl [ukončen](https://github.com/pytorch/pytorch/issues/138506) od verze 2.5.1. Stále však existuje [komunitní verze PyTorch](https://anaconda.org/conda-forge/pytorch), kterou používám a která podporuje instalaci přes Anacondu včetně podpory GPU.

**V konzoli Anaconda spusťte následující příkazy pro instalaci:**

```bash
conda create -n my_env
conda activate my_env
conda install conda-forge::pytorch
```

**Ověření instalace:**

```python
python -c "import torch; print(torch.__version__); print(torch.__path__)"
# 2.7.1+cu128 => verze se může lišit
```

✅ *Pokud instalace proběhne bez chyb a tento příkaz vrátí očekávanou verzi, mělo by být vše připraveno k použití.*

> ⚠️ **Známé problémy a jejich řešení:**
>
> Při instalaci se může objevit [chyba s DLL](https://discuss.pytorch.org/t/importerror-dll-load-failed-while-importing-c-das-angegebene-modul-wurde-nicht-gefunden-the-specified-module-can-not-be-found/217569), která je poměrně častá. Tento problém jsem vyřešil instalací předchozí verze PyTorch z komunitní distribuce.
>
> Aktuální verze PyTorch je *2.8.0*, ale instalátor této verze je stále ve verzi *2.7.1*. Tento problém nastal, když jsem se pokusil nainstalovat současnou verzi. Pokud bude k dispozici novější verze, nebo instalátor bude v stejné verzi jako knihovna samotná, doporučuji ji vyzkoušet bez ohledu na verzi. V případě potíží lze konkrétní verzi nainstalovat pomocí pip:
>
> ```bash
> pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
> ```

## Použité datasety

Svoboda, R., et al. (2024). *Natural Gas Consumption Forecasting System*.  
Dataset AI VŠB – Natural Gas Forecasting. Dostupné z tohoto [odkazu](https://ai.vsb.cz/natural-gas-forecasting).  
DOI: [10.1016/j.energy.2020.119430](https://doi.org/10.1016/j.energy.2020.119430)
