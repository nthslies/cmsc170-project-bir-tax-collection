# imports
import tkinter as tk
from tkinter.filedialog import askopenfilename
import zipfile
import csv
import io
import re
import numpy as np
import math

# function to read uploaded zip file
def read_collection(zip_path = None):
    # initialize window in order to open file
    if zip_path is None:
        root = tk.Tk()
        root.withdraw()
        zip_path = askopenfilename(
            title = "Select BIR Tax Collection ZIP File",
            filetypes = [
                ("ZIP files", "*.zip"),
                ("All files", "*.*")
            ]
        )
        root.destroy()

    # if no file selected
    if not zip_path:
        print("No ZIP file selected.")
        return None
    print(f"{zip_path} successfully loaded!")

    # for storing monthly collections
    collections = {}

    # maps month to equivalent number
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    try:
        # opening a zip file
        with zipfile.ZipFile(zip_path, 'r') as zip:
            for filename in sorted(zip.namelist()):
                if 'bir_collection' in filename.lower() and filename.lower().endswith('.csv'):
                    # for extracting the year from filename
                    year_match = re.search(r'(202[0-4])', filename)
                    if not year_match:
                        continue
                    year = int(year_match.group())

                    # opening each file in zip
                    with zip.open(filename) as f:
                        text = io.TextIOWrapper(f, encoding = 'utf-8-sig')
                        reader = csv.DictReader(text)
                        # ensures header exists
                        header = reader.fieldnames
                        if not header:
                            continue
                        # cleaning the headers
                        cleaned_header = {h: h.strip().lower() for h in header if h} 
                        # gets regions
                        region_column = None
                        for h in header:
                            if  h and h.strip().lower() == 'region':
                                region_column = h
                                break

                        # maps month to its equivalent number
                        active_month = {origin: months[clean] for origin, clean in cleaned_header.items() if clean in months}                    
                        # ensures required columns exists
                        if len(active_month) != 12 or not region_column:
                            continue

                        # parse each row
                        for row in reader:
                            region = row.get(region_column, '')
                            if region is None:
                                continue
                            region = region.strip()
                            if not region or 'total' in region.lower():
                                continue

                            # loops in each active month
                            for column_name, month_number in active_month.items():
                                raw_value = row.get(column_name, '')
                                if not raw_value:
                                    continue
                                cleaned_value = str(raw_value).replace(',', '').strip()

                                if not cleaned_value or cleaned_value == "-" or cleaned_value.lower() == "nan" or cleaned_value.lower() == "n/a":
                                    continue

                                try:
                                    revenue = float(cleaned_value)
                                    # skips nan and infinity values
                                    if not math.isfinite(revenue):
                                        continue
                                    key = (region, month_number, year)
                                    if key not in collections:
                                        collections[key] = 0.0
                                    # adds collections from same region
                                    collections[key] += revenue
                                except ValueError:
                                    continue

    except (zipfile.BadZipFile, OSError) as error:
        print(f"Error opening ZIP file: {error}")
        return None

    if not collections:
        print("No valid collection data found.")
        return None

    return collections

# gets date of previous month with year transitions
def previous_month(year, month, steps = 1):
    index = year * 12 + month - 1 - steps
    return index // 12, index % 12 + 1

# calculates month-over-month growth for a region
def monthly_growth(collections, region, month, year):
    previous_year, previous_month_num = previous_month(year, month)
    current = collections.get((region, month, year))
    previous = collections.get((region, previous_month_num, previous_year))
    if current is None or previous is None or previous <= 0:
        return None
    return (current - previous) / previous

# functions for loading data needed
def load_data(zip_path = None, return_years = False):
    collections = read_collection(zip_path)
    if collections is None:
        if return_years:
            return None, None, None, None, None
        return None, None, None, None
    # sorting the regions
    sorted_region = sorted(set(region for region, month, year in collections))
    # maps region to index
    region_to_id = {region: index for index, region in enumerate(sorted_region)}
    # initializes train and test data
    x_train, y_train, x_test, y_test = [], [], [], []
    year_train = [] # year of each training row, for the validation folds

    # -- CHANGED ONE-HOT TO BEHAVIORAL 
    # bali iimplement natin yun from k-means
    feature_names = [
        "region_mean", "region_annual_growth", "region_volatility", 
        "month_sin", "month_cos", "previous_month", "previous_3month_mean",
        "previous_3month_volatility", "previous_month_mom_growth", "historical_median_mom_growth"
    ]
    csv_rows = []
    
    # Pre-calculate behavioral features for all regions
    region_profiles = {}
    for r in sorted_region:
        monthly_values = [collections[k] for k in collections if k[0] == r]
        yearly_totals = {y: sum(collections[(r, m, y)] for m in range(1, 13) if (r, m, y) in collections) for y in range(2020, 2025)}
        mean_rev = float(np.mean(monthly_values)) if monthly_values else 0.0
        volatility = float(np.std(monthly_values)) / abs(mean_rev) if mean_rev != 0 else 0.0
        growth_rates = [((yearly_totals[y] - yearly_totals[y-1]) / yearly_totals[y-1]) for y in range(2021, 2025) if yearly_totals.get(y-1, 0) != 0]
        avg_growth = float(np.mean(growth_rates)) if growth_rates else 0.0
        region_profiles[r] = [mean_rev, avg_growth, volatility]

    for region in sorted_region:
        earlier_growth = []
        for year in range(2020, 2025):
            for month in range(1, 13):
                current_revenue = collections.get((region, month, year))
                if current_revenue is None:
                    continue

                target_growth = monthly_growth(collections, region, month, year)
                previous_year, previous_month_num = previous_month(year, month)
                previous_growth = monthly_growth(collections, region, previous_month_num, previous_year)
                # gets previous 3 months
                prior_keys = []
                for steps in (1, 2, 3):
                    prior_year, prior_month = previous_month(year, month, steps)
                    prior_keys.append((region, prior_month, prior_year))
                # starts with April 2020
                if (year > 2020 or month >= 4) and target_growth is not None and previous_growth is not None and earlier_growth and all(key in collections for key in prior_keys):
                    previous_values = [collections[key] for key in prior_keys]
                    previous_mean = float(np.mean(previous_values))
                    previous_volatility = (float(np.std(previous_values)) / max(abs(previous_mean), 1e-9))
                    historical_median = float(np.median(earlier_growth))

                    # 3 behavioral metrics set
                    region_features = region_profiles[region]
                    
                    # to represent months as yearly cycle
                    month_sin = math.sin(2 * math.pi * month / 12)
                    month_cos = math.cos(2 * math.pi * month / 12)
                    features = region_features + [ month_sin, month_cos, previous_values[0],
                        previous_mean, previous_volatility, previous_growth, historical_median]
                    # marks 1 if greater than historical median
                    label = int(target_growth > historical_median)
                    # for identifying if training or test data
                    if year <= 2023:
                        x_train.append(features)
                        y_train.append(label)
                        year_train.append(year)
                        split = "train"
                    elif year == 2024:
                        x_test.append(features)
                        y_test.append(label)
                        split = "test"
                    csv_rows.append([split, region, year, month, f"{previous_year}-{previous_month_num:02d}",
                                     *features, label])

                if target_growth is not None:
                    earlier_growth.append(target_growth)

    # checks if train or test data is empty
    if not x_train or not x_test:
        print("Training or testing data is empty.")
        if return_years:
            return None, None, None, None, None
        return None, None, None, None

    csv_rows.sort(key = lambda row: (row[2], row[3], row[1]))
    with open("cleaned_data.csv", "w", newline = "", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "region", "year", "month", "basis", *feature_names, "label"])
        writer.writerows(csv_rows)
    
    # converts to arrays
    x_train = np.array(x_train, dtype = float)
    y_train = np.array(y_train, dtype = int)
    x_test = np.array(x_test, dtype = float)
    y_test = np.array(y_test, dtype = int)

    # prints info about dataset
    print(f"\nNumber of Regions: {len(sorted_region)}")
    print(f"Training Records (April 2020 - 2023): {len(x_train)}")
    print(f"Testing Records (2024): {len(x_test)}")
    print(f"Number of Features: {x_train.shape[1]}")
    print("File saved as 'cleaned_data.csv'")
    
    # the years are only returned when asked, so the other files that use load_data still work
    if return_years:
        return x_train, y_train, x_test, y_test, year_train
    return(x_train, y_train, x_test, y_test)

def load_clustering(zip_path = None):
    collections = read_collection(zip_path)
    if collections is None:
        return None, None
    # sorting the regions
    sorted_region = sorted(set(region for region, month, year in collections))
    # gets all regions
    x_regions = []
    region_names = []
    for region in sorted_region:
        monthly_values = []
        yearly_totals = {}
        # collects values from each month from 2020 to 2024
        for year in range(2020, 2025):
            year_values = []
            for month in range(1, 13):
                key = (region, month, year)
                if key in collections:
                    revenue = collections[key]
                    monthly_values.append(revenue)
                    year_values.append(revenue)
            # ensures all 12 months exists for getting sum of year
            if len(year_values) == 12:
                yearly_totals[year] = sum(year_values)

        # ensures region has complete data
        if len(monthly_values) != 60 or len(yearly_totals) != 5:
            print(f"Skipped {region}: incomplete collection history.")
            continue

        mean_monthly_collection = np.mean(monthly_values) # means
        growth_rates = [] # annual growth rate
        for year in range(2021, 2025):
            previous_total = yearly_totals[year - 1]
            current_total = yearly_totals[year]
            # handles division by zero
            if previous_total != 0:
                growth = ((current_total - previous_total) / previous_total)
                growth_rates.append(growth)

        # ensures theres 4 growth rates
        if len(growth_rates) != 4:
            print(f"Skipped {region}: growth rate unavailable.")
            continue
        average_annual_growth_rate = np.mean(growth_rates)

        if mean_monthly_collection == 0:
            print(f"Skipped {region}: zero mean collection.")
            continue
        collection_volatility = (np.std(monthly_values) / abs(mean_monthly_collection))
        # combine clustering features
        features = [mean_monthly_collection, average_annual_growth_rate, collection_volatility]
        x_regions.append(features)
        region_names.append(region)

    if not x_regions:
        print("No valid clustering data found.")
        return None, None
    # converts to array
    x_regions = np.array(x_regions, dtype = float)
    # prints info
    print(f"Number of Regions: {len(region_names)}")
    print(f"Number of Features: {x_regions.shape[1]}")

    return x_regions, region_names

if __name__ == "__main__":
    # select ZIP
    root = tk.Tk()
    root.withdraw()
    selected_zip = askopenfilename(
        title = "Select BIR Tax Collection ZIP File",
        filetypes = [
            ("ZIP files", "*.zip"), 
            ("All files", "*.*")
        ]
    )
    root.destroy()
    if not selected_zip:
        print("No ZIP file selected.")
        raise SystemExit

    # supervised
    x_train, y_train, x_test, y_test = load_data(selected_zip)
    if x_train is not None:
        # indicator that data loaded
        print("\nData Loading Completed!")

    # unsupervised
    x_regions, region_names = load_clustering(selected_zip)
    if x_regions is not None:
        # indicator that data loaded
        print("\nData Loading Completed!")