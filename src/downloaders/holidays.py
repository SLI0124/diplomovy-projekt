import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import re


MONTHS_TO_NUMBER = {
    "ledna": 1,
    "února": 2,
    "března": 3,
    "dubna": 4,
    "května": 5,
    "června": 6,
    "července": 7,
    "srpna": 8,
    "září": 9,
    "října": 10,
    "listopadu": 11,
    "prosince": 12,
}


def load_dataset(file_path, separator=";"):
    if not Path(file_path).is_file():
        print(f"File {file_path} does not exist.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, sep=separator)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()


def compare_dfs(name, df1, df2):
    if df1.equals(df2):
        print(f"{name}: DataFrames are identical.")
    else:
        print(f"{name}: DataFrames are different.")
        # Show rows in df1 not in df2
        diff1 = pd.concat([df1, df2, df2]).drop_duplicates(keep=False)
        if not diff1.empty:
            print("Rows in first DataFrame not in second:")
            print(diff1)
        # Show rows in df2 not in df1
        diff2 = pd.concat([df2, df1, df1]).drop_duplicates(keep=False)
        if not diff2.empty:
            print("Rows in second DataFrame not in first:")
            print(diff2)


def scrape_czech_holidays(year):
    url = f"https://www.kurzy.cz/kalendar/statni-svatky/{year}/"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data for year {year}")
        return pd.DataFrame()
    soup = BeautifulSoup(response.text, "html.parser")
    kalendar_div = soup.find("div", id="kalendar")
    year_div = None
    if kalendar_div:
        for title_variant in (f"Státní svátky v roce {year}", f"Státní svátky {year}"):
            year_div = kalendar_div.find("div", attrs={"title": title_variant})
            if year_div:
                break
        if year_div is None:
            year_div = kalendar_div.find(
                "div", attrs={"title": re.compile(rf".*{year}.*")}
            )
        if year_div is None:
            year_div = kalendar_div.find(
                "div", id=re.compile(rf"^Year_statni-svatky.*{year}")
            )
    if year_div is None:
        print(f"No div found for year {year}")
        return pd.DataFrame()
    data_div = year_div.find("div", class_="data only-list")
    if not data_div:
        print("No data div found")
        return pd.DataFrame()
    day_divs = data_div.find_all("div", class_="den")
    holidays = []
    for day in day_divs:
        day_span = day.find("span")
        if day_span:
            day_text = day_span.get_text(strip=True)
            match = re.search(r"(\d{1,2})\.?\s+([^\s]+)\s+\d{4}", day_text)
            if match:
                day_number = int(match.group(1))
                month_name_raw = match.group(2)
                month_key = month_name_raw.strip().lower()
                # map month name to number using months_to_number defined elsewhere in the notebook
                month_num = MONTHS_TO_NUMBER.get(month_key)
                # try fuzzy match if exact key not found
                if month_num is None:
                    for k, v in MONTHS_TO_NUMBER.items():
                        if month_key.startswith(k) or k.startswith(month_key):
                            month_num = v
                            break
                if month_num is None:
                    print(f"Unknown month name: {month_name_raw} (skipping)")
                    continue
                holiday_name = day.find("a", href=True).get_text(strip=True)
                holidays.append(
                    {
                        "day": day_number,
                        "month": month_num,
                        "holiday_name": holiday_name,
                    }
                )
    return pd.DataFrame(holidays)


def main():
    year = 2015
    scrapped_holidays_df = scrape_czech_holidays(year)

    original_df = load_dataset("../../data/original/ppnet_metar.csv")
    original_df_before_holiday = original_df[
        (original_df["before_holiday"] == 1) & (original_df["year"] == year)
    ]

    original_df_before_holiday = (
        original_df_before_holiday.groupby(["year", "month", "day"])
        .first()
        .reset_index()
    )

    original_df_before_holiday = original_df_before_holiday[
        ["year", "month", "day", "before_holiday"]
    ]

    # we will compare only month and day columns since that is only relevant info for dataset
    temp_original_df = original_df_before_holiday.copy()
    temp_original_df = temp_original_df[["month", "day"]]

    temp_scraped_df = scrapped_holidays_df.copy()
    temp_scraped_df = temp_scraped_df[["month", "day"]]

    compare_dfs("Holidays before holiday", temp_scraped_df, temp_original_df)


if __name__ == "__main__":
    main()
