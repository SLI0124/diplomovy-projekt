# **Diplomový projekt**

*Cílem této práce je představit a popsat metody předpovídání časových řad s využitím hlubokých neuronových sítí.*

V tomto repozitáři najdete ukázky a popisy implementace metod pro předpovídání časových řad s využitím hlubokého učení. Zaměřuji se zde na praktickou stránku práce s kódem, jeho ověřování na reálných datech a srovnání dosažených výsledků.

## ⚙️ **Instalace**

Pro implementaci projektu byla zvolena knihovna **PyTorch**, která nabízí výbornou podporu pro práci s grafickými akcelerátory. Instalace a konfigurace této knihovny může být někdy složitější, proto **doporučuji řídit se oficiální dokumentací** dostupnou [zde](https://pytorch.org/get-started/locally/).

### 🐍 **Anaconda**

PyTorch byl dlouhou dobu dostupný k instalaci přes **Anacondu**, avšak tento způsob instalace byl [ukončen](https://github.com/pytorch/pytorch/issues/138506) od verze 2.5.1. Aktuálně je doporučeno mít nainstalovaný runtime pro **CUDA**.

Stále však existuje [komunitní verze PyTorch](https://anaconda.org/conda-forge/pytorch), kterou používám a která podporuje instalaci přes Anacondu včetně podpory GPU. Během instalace jsem narazil na [problém](https://discuss.pytorch.org/t/importerror-dll-load-failed-while-importing-c-das-angegebene-modul-wurde-nicht-gefunden-the-specified-module-can-not-be-found/217569), který přetrvává a je poměrně častý. Vyřešil jsem jej instalací předposlední verze PyTorch z komunitní distribuce, dostupné [zde](https://pytorch.org/get-started/previous-versions/). Více během instalace.

**V konzoli Anaconda spusťte následující příkazy pro instalaci:**

```python
conda create -n master-thesis
conda activate master-thesis
conda install conda-forge::pytorch
```

```python
python -c "import torch; print(torch.__version__); print(torch.__path__)"
# 2.7.1+cu128
```

✅ *Pokud instalace proběhne bez chyb a tento příkaz vrátí očekávanou verzi, mělo by být vše připraveno k použití.*

> ⚠️ **Varování pro Anacondu:**
>
> V případě problémů je nutné doinstalovat další závislosti. Jak již bylo zmíněno, tato verze knihovny nemusí být plně funkční na všech systémech v rámci Anaconda prostředí. Proto doporučuji zkusit novější verzi, pokud je k dispozici. Během instalace je nejnovější verze *2.8.0*, ale podařilo se mi rozběhnout verzi *2.7.1*.

Stačí spustit následující příkaz:

```python
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```
