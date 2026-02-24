# Diplomový projekt

*Cílem této práce je představit a popsat metody předpovídání časových řad s využitím hlubokých neuronových sítí.*

V tomto repozitáři najdete ukázky a popisy implementace metod pro předpovídání časových řad s využitím hlubokého učení. Zaměřuji se zde na praktickou stránku práce s kódem, jeho ověřování na reálných datech a srovnání dosažených výsledků. Některé natrénované modely mohou být poměrně velké, a proto nebudou součástí tohoto repozitáře.

Součástí tohoto repozitáře je také sběr dat, na kterých je vše postaveno.

## ⚙️ Instalace

### Předpoklady

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **CUDA Runtime/Driver** (pro GPU podporu) - viz [oficiální instalační průvodce NVIDIA](https://developer.nvidia.com/cuda-downloads)

### 📦 Instalace pomocí uv (volba backendu)

Projekt je nastavený tak, aby se PyTorch backend vybíral přes `pyproject.toml` extras:

- `cpu`
- `cu126`
- `cu130`

#### CPU varianta

```bash
uv sync --extra cpu
```

#### CUDA 12.6 varianta

```bash
uv sync --extra cu126
```

#### CUDA 13.0 varianta

```bash
uv sync --extra cu130
```

> 💡 Doporučení: používej vždy jen jednu variantu (`cpu` / `cu126` / `cu130`) podle cílového prostředí.

### ✅ Ověření instalace

```bash
uv run python tools/check_gpu.py
```

### 📚 Reference

- uv + PyTorch integrace: [https://docs.astral.sh/uv/guides/integration/pytorch/](https://docs.astral.sh/uv/guides/integration/pytorch/)
- PyTorch instalace (oficiální): [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### ➕ Do budoucna: přidání dalšího backendu/indexu

Pokud budeš chtít přidat další variantu (např. `cu128`), uprav:

1. `[project.optional-dependencies]` (nové extra)
2. `[tool.uv.sources]` (mapování `torch` na index + extra)
3. `[[tool.uv.index]]` (nový PyTorch index)
4. `[tool.uv].conflicts` (doplň nové extra do jednoho seznamu konfliktů)
5. poté spusť `uv lock`
