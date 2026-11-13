# Milestone 3 K-Means Clustering on BIR Tax Collection Data

import os
import csv
import math
import random
import matplotlib.pyplot as plt
from filereader import read_collection, monthly_growth

RESTARTS = 100 # how many times k-means is run for each k (the best run is kept)

# asks which data to cluster
def ask_mode():
  while True:
    mode = input("Cluster 1 = one point per region, 2 = one point per region and year (more points): ").strip()
    if mode == "1" or mode == "2":
      return int(mode)
    print("Enter 1 or 2!")

# asks for the number of clusters, 0 means pick the best k
def ask_k(max_k):
  while True:
    try:
      k = int(input(f"Enter the number of clusters, k (0 to pick the best k from 2 to {max_k}): "))
      if k == 0 or 2 <= k <= max_k:
        return k
      print(f"k must be 0 or from 2 to {max_k}!")
    except ValueError:
      print("k must be a whole number!")

# short name of a region, like BARMM, CAR, I, II, IV-A (a year at the end is kept)
def short_name(name):
  year = ""
  last_word = name.split(" ")[-1]
  if last_word.isdigit() and len(last_word) == 4:
    year = " " + last_word
    name = name[:-5]
  if name.startswith("Region "):
    short = name.split(" ")[1]
  elif "(" in name:
    short = name.split("(")[-1].rstrip(")")
  else:
    short = name.replace(" Region", "")
  return short + year

# gets the 12 monthly values of each year of a region, years with a missing month are left out
def get_region_years(collections, region):
  years = {}
  for year in range(2020, 2025):
    values = []
    for month in range(1, 13):
      key = (region, month, year)
      if key in collections:
        values.append(collections[key])
    if len(values) == 12:
      years[year] = values
  return years

# median of a list of values
def median(values):
  ordered = sorted(values)
  middle = len(ordered) // 2
  if len(ordered) % 2 == 1:
    return ordered[middle]
  return (ordered[middle - 1] + ordered[middle]) / 2

# volatility, std of the monthly values over their mean
def volatility_of(values):
  mean_value = sum(values) / len(values)
  variance = sum((value - mean_value) ** 2 for value in values) / len(values)
  return math.sqrt(variance) / mean_value

# month-over-month growth rates of a region for the given years, uses monthly_growth of filereader.py
def growth_rates_of(collections, region, years):
  rates = []
  for year in years:
    for month in range(1, 13):
      rate = monthly_growth(collections, region, month, year)
      if rate is not None: # the first month of 2020 has no month before it
        rates.append(rate)
  return rates

# total collection of one calendar quarter (1 to 4), None if a month is missing
def quarter_total(collections, region, year, quarter):
  values = [collections.get((region, month, year)) for month in range(3 * quarter - 2, 3 * quarter + 1)]
  if any(value is None for value in values):
    return None
  return sum(values)

# quarter-over-quarter growth rates of a region for the given years
# 2020 starts at April: April to June is compared with January to March, so the 2020 data is used too
def quarter_growth_rates_of(collections, region, years):
  rates = []
  for year in years:
    for quarter in range(1, 5):
      current = quarter_total(collections, region, year, quarter)
      if quarter == 1:
        before = quarter_total(collections, region, year - 1, 4)
      else:
        before = quarter_total(collections, region, year, quarter - 1)
      if current is not None and before is not None and before > 0:
        rates.append((current - before) / before)
  return rates

# one point per region, using the collections of 2020 to 2024
def region_data(collections):
  names = []
  rows = []
  for region in sorted(set(region for region, month, year in collections)):
    years = get_region_years(collections, region)
    if len(years) != 5:
      print(f"Skipped {region}: incomplete collection history.")
      continue
    monthly_values = []
    for year in range(2020, 2025):
      monthly_values += years[year]
    if min(monthly_values) <= 0:
      print(f"Skipped {region}: zero collection.")
      continue
    names.append(region)
    rows.append([median(monthly_values), median(growth_rates_of(collections, region, range(2020, 2025))),
                 median(quarter_growth_rates_of(collections, region, range(2020, 2025))), volatility_of(monthly_values)])
  return names, rows

# one point per region and year (2020 to 2024), so there are more points to cluster
def region_year_data(collections):
  names = []
  rows = []
  for region in sorted(set(region for region, month, year in collections)):
    years = get_region_years(collections, region)
    if len(years) != 5:
      print(f"Skipped {region}: incomplete collection history.")
      continue
    if min(min(years[year]) for year in years) <= 0:
      print(f"Skipped {region}: zero collection.")
      continue
    for year in range(2020, 2025):
      names.append(f"{region} {year}")
      rows.append([median(years[year]), median(growth_rates_of(collections, region, [year])),
                   median(quarter_growth_rates_of(collections, region, [year])), volatility_of(years[year])])
  return names, rows

# makes the features of each point
def make_features(rows):
  features = []
  for median_monthly, mom_growth, qoq_growth, volatility in rows:
    features.append([math.log10(median_monthly), mom_growth, qoq_growth, volatility]) # log so big regions do not take over
  return features

# mean and std of each column
def column_stats(matrix):
  n = len(matrix)
  means = [sum(row[j] for row in matrix) / n for j in range(len(matrix[0]))]
  stds = []
  for j in range(len(means)):
    variance = sum((row[j] - means[j]) ** 2 for row in matrix) / n
    stds.append(math.sqrt(variance))
  return means, stds

# z-score, a std of 0 is changed to 1
def standardize(matrix, means, stds):
  scaled = []
  for row in matrix:
    scaled.append([(row[j] - means[j]) / (stds[j] if stds[j] > 0 else 1.0) for j in range(len(row))])
  return scaled

# prints the data that the algorithm sees
def print_preprocessed(names, rows, points):
  print("\n-----------PREPROCESSED DATA-----------")
  print(f"Number of points: {len(rows)}")
  print("Features: log10 of the median monthly collection, median month-over-month growth, median quarter-over-quarter growth, volatility (all standardized)")
  print(f"{'Point':<14}{'Median Monthly':<16}{'MoM Growth (%)':<16}{'QoQ Growth (%)':<16}{'Volatility':<12}{'z Size':<9}{'z MoM':<9}{'z QoQ':<9}{'z Volat.':<9}")
  for i in range(len(rows)):
    print(f"{short_name(names[i]):<14}{rows[i][0]:<16,.2f}{rows[i][1] * 100:<16.2f}{rows[i][2] * 100:<16.2f}{rows[i][3]:<12.3f}"
          f"{points[i][0]:<9.2f}{points[i][1]:<9.2f}{points[i][2]:<9.2f}{points[i][3]:<9.2f}")

# squared distance of two points
def squared_distance(point, centroid):
  total = 0
  for j in range(len(point)):
    total += (point[j] - centroid[j]) ** 2
  return total

# picks the starting centroids, far points are more likely to be picked (k-means++)
def start_centroids(points, k, random_gen):
  centroids = [list(random_gen.choice(points))]
  while len(centroids) < k:
    distances = [min(squared_distance(point, centroid) for centroid in centroids) for point in points]
    total = sum(distances)
    if total == 0: # every point is already a centroid
      centroids.append(list(random_gen.choice(points)))
      continue
    pick = random_gen.random() * total
    running = 0
    chosen = len(points) - 1
    for i in range(len(points)):
      running += distances[i]
      if running >= pick:
        chosen = i
        break
    centroids.append(list(points[chosen]))
  return centroids

# finds the nearest centroid
def nearest_centroid(point, centroids):
  best = 0
  best_distance = squared_distance(point, centroids[0])
  for c in range(1, len(centroids)):
    distance = squared_distance(point, centroids[c])
    if distance < best_distance: # strictly smaller, so ties go to the first centroid
      best = c
      best_distance = distance
  return best

# moves each centroid to the mean of its points
def update_centroids(points, labels, centroids):
  new_centroids = []
  for c in range(len(centroids)):
    members = [points[i] for i in range(len(points)) if labels[i] == c]
    if len(members) == 0:
      farthest = 0 # empty cluster, takes the point that is farthest from its centroid
      farthest_distance = -1
      for i in range(len(points)):
        distance = squared_distance(points[i], centroids[labels[i]])
        if distance > farthest_distance:
          farthest = i
          farthest_distance = distance
      new_centroids.append(list(points[farthest]))
      continue
    new_centroids.append([sum(column) / len(members) for column in zip(*members)])
  return new_centroids

# inertia, total squared distance to the own centroid
def compute_inertia(points, labels, centroids):
  total = 0
  for i in range(len(points)):
    total += squared_distance(points[i], centroids[labels[i]])
  return total

# one run of k-means
def kmeans_once(points, k, random_gen, max_iterations=300):
  centroids = start_centroids(points, k, random_gen)
  labels = None
  iteration = 0
  while iteration < max_iterations:
    iteration += 1
    new_labels = [nearest_centroid(point, centroids) for point in points]
    if new_labels == labels: # nothing moved, so stop
      break
    labels = new_labels
    centroids = update_centroids(points, labels, centroids)
  return {"labels": labels, "centroids": centroids, "iterations": iteration,
          "inertia": compute_inertia(points, labels, centroids)}

# runs k-means many times, keeps the lowest inertia
def kmeans(points, k, random_gen, restarts=RESTARTS):
  best = None
  all_inertias = []
  for _ in range(restarts):
    run = kmeans_once(points, k, random_gen)
    all_inertias.append(run["inertia"])
    if best is None or run["inertia"] < best["inertia"]:
      best = run
  best["all_inertias"] = all_inertias
  return best

# distance of every pair of points
def pairwise_distances(points):
  n = len(points)
  distances = [[0.0] * n for _ in range(n)]
  for i in range(n):
    for j in range(i + 1, n):
      distances[i][j] = math.sqrt(squared_distance(points[i], points[j]))
      distances[j][i] = distances[i][j]
  return distances

# silhouette of each point
def silhouette_scores(distances, labels, k):
  members = [[] for _ in range(k)]
  for i in range(len(labels)):
    members[labels[i]].append(i)
  scores = []
  for i in range(len(labels)):
    own = labels[i]
    if len(members[own]) == 1: # alone in its cluster
      scores.append(0.0)
      continue
    a = sum(distances[i][j] for j in members[own] if j != i) / (len(members[own]) - 1)
    b = None # distance to the nearest other cluster
    for c in range(k):
      if c == own or len(members[c]) == 0:
        continue
      mean_distance = sum(distances[i][j] for j in members[c]) / len(members[c])
      if b is None or mean_distance < b:
        b = mean_distance
    if b is None or max(a, b) == 0:
      scores.append(0.0)
    else:
      scores.append((b - a) / max(a, b))
  return scores

# number of points in each cluster
def cluster_sizes(labels, k):
  sizes = [0] * k
  for label in labels:
    sizes[label] += 1
  return sizes

# tries every k from 2 to max_k, one dictionary of results for each k
def try_all_k(points, distances, random_gen, max_k):
  results = []
  for k in range(2, max_k + 1):
    run = kmeans(points, k, random_gen)
    scores = silhouette_scores(distances, run["labels"], k)
    run["k"] = k
    run["silhouette"] = sum(scores) / len(scores)
    run["smallest"] = min(cluster_sizes(run["labels"], k))
    results.append(run)
  return results

# finds the elbow, the k farthest from the line between the first and the last point
def find_elbow(results):
  ks = [result["k"] for result in results]
  inertias = [result["inertia"] for result in results]
  x = [(k - ks[0]) / (ks[-1] - ks[0]) for k in ks] # both axes are scaled from 0 to 1
  y = [(inertia - inertias[-1]) / (inertias[0] - inertias[-1]) for inertia in inertias]
  best_k = ks[0]
  best_distance = -1
  for i in range(len(ks)):
    distance = abs(x[i] + y[i] - 1) / math.sqrt(2) # distance to the line from (0, 1) to (1, 0)
    if distance > best_distance:
      best_k = ks[i]
      best_distance = distance
  return best_k

# balances silhouette and inertia, uses the elbow k if its silhouette is close to the best
def choose_k(results, elbow_k):
  valid = [result for result in results if result["smallest"] >= 2] # a cluster of 1 point is not a group
  if len(valid) == 0:
    valid = results
  best = valid[0]
  for result in valid:
    if result["silhouette"] > best["silhouette"]:
      best = result
  chosen = best
  for result in valid:
    if result["k"] == elbow_k and best["silhouette"] - result["silhouette"] <= 0.05:
      chosen = result
  return best["k"], chosen["k"]

# describes a cluster as high, average, or low in each standardized feature
def describe_cluster(center):
  words = []
  for name, value in zip(["size", "MoM growth", "QoQ growth", "volatility"], center):
    if value > 0.5:
      level = "high"
    elif value < -0.5:
      level = "low"
    else:
      level = "average"
    words.append(f"{name} {level}")
  return ", ".join(words)

# gets the profile of each cluster from the original values
def cluster_profiles(rows, final, k):
  profiles = []
  for c in range(k):
    members = [i for i in range(len(rows)) if final["labels"][i] == c]
    profile = {"members": members, "meaning": describe_cluster(final["centroids"][c])}
    for position, name in enumerate(["median_monthly", "growth", "quarter_growth", "volatility"]):
      profile[name] = median([rows[i][position] for i in members]) if len(members) > 0 else 0.0
    profiles.append(profile)
  return profiles

# prints the profile of each cluster, what the values mean, and the points of each cluster
def display_clusters(names, profiles):
  print("\n-----------CLUSTER PROFILES-----------")
  print(f"{'Cluster':<9}{'# Points':<10}{'Median Monthly (M PHP)':<24}{'Median MoM Growth (%)':<23}{'Median QoQ Growth (%)':<23}{'Volatility':<12}{'Meaning'}")
  for c, profile in enumerate(profiles):
    print(f"{c:<9}{len(profile['members']):<10}{profile['median_monthly']:<24,.2f}{profile['growth'] * 100:<23.2f}{profile['quarter_growth'] * 100:<23.2f}{profile['volatility']:<12.3f}{profile['meaning']}")
  print("\n-----------POINTS PER CLUSTER-----------")
  for c, profile in enumerate(profiles):
    print(f"Cluster {c}: " + ", ".join(short_name(names[i]) for i in profile["members"]))

# saves a table as a csv, the file must not be open in another program
def save_table(path, header, table_rows):
  try:
    with open(path, "w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerows(table_rows)
  except PermissionError:
    print(f"Could not save {path}, close the file if it is open.")

# saves a figure, if the file cannot be written (for example it is open in a viewer) it is saved under a new name
def save_figure(figure, path):
  try:
    figure.savefig(path, dpi=150)
  except OSError as error:
    fallback = path[:-4] + "_new.png"
    print(f"Could not save {path} ({error}). Close the file if it is open in another program.")
    try:
      figure.savefig(fallback, dpi=150)
      print(f"Saved it as {fallback} instead.")
    except OSError as second_error:
      print(f"Could not save {fallback} either ({second_error}).")

# saves the elbow graph and the silhouette graph
def plot_k_graphs(results, elbow_k, chosen_k, folder, tag):
  ks = [result["k"] for result in results]
  figure, axis = plt.subplots(figsize=(6, 4))
  axis.plot(ks, [result["inertia"] for result in results], marker="o")
  axis.axvline(elbow_k, color="gray", linestyle="--", label=f"elbow k = {elbow_k}")
  axis.set_title("Elbow method")
  axis.set_xlabel("k")
  axis.set_ylabel("Inertia")
  axis.legend()
  figure.tight_layout()
  save_figure(figure, f"{folder}/elbow_method_{tag}.png")
  plt.close(figure)
  figure, axis = plt.subplots(figsize=(6, 4))
  axis.plot(ks, [result["silhouette"] for result in results], marker="o", color="tab:orange")
  axis.axvline(chosen_k, color="gray", linestyle="--", label=f"chosen k = {chosen_k}")
  axis.set_title("Silhouette score")
  axis.set_xlabel("k")
  axis.set_ylabel("Mean silhouette")
  axis.legend()
  figure.tight_layout()
  save_figure(figure, f"{folder}/silhouette_by_k_{tag}.png")
  plt.close(figure)

# saves the bar charts of the three values of each cluster
def plot_profiles(profiles, folder, tag):
  names = [f"Cluster {c}\n(n={len(profiles[c]['members'])})" for c in range(len(profiles))]
  figure, axes = plt.subplots(1, 4, figsize=(16, 4))
  axes[0].bar(names, [profile["median_monthly"] for profile in profiles], color="tab:blue")
  axes[0].set_yscale("log") # the clusters have very different sizes
  axes[0].set_title("Median monthly collection")
  axes[0].set_ylabel("Million PHP (log scale)")
  axes[1].bar(names, [profile["growth"] * 100 for profile in profiles], color="tab:green")
  axes[1].set_title("Median month-over-month growth")
  axes[1].set_ylabel("Percent")
  axes[2].bar(names, [profile["quarter_growth"] * 100 for profile in profiles], color="tab:olive")
  axes[2].set_title("Median quarter-over-quarter growth")
  axes[2].set_ylabel("Percent")
  axes[3].bar(names, [profile["volatility"] for profile in profiles], color="tab:red")
  axes[3].set_title("Volatility")
  axes[3].set_ylabel("Std / mean of monthly collections")
  figure.tight_layout()
  save_figure(figure, f"{folder}/cluster_profiles_{tag}.png")
  plt.close(figure)

# saves the scatter plots of the points, colored by cluster (size against each other feature)
def plot_scatter(names, rows, labels, k, folder, tag):
  panels = [(1, 100, "Median MoM growth (%)", "Size and MoM growth"),
            (2, 100, "Median QoQ growth (%)", "Size and QoQ growth"),
            (3, 1, "Volatility", "Size and volatility")]
  figure, axes = plt.subplots(1, 3, figsize=(17, 5))
  for position, (column, scale, label, title) in enumerate(panels):
    for c in range(k):
      members = [i for i in range(len(rows)) if labels[i] == c]
      x = [math.log10(rows[i][0]) for i in members]
      y = [rows[i][column] * scale for i in members]
      axes[position].scatter(x, y, label=f"Cluster {c}", color=plt.cm.tab10(c % 10))
      if len(rows) <= 30:
        for i in members:
          axes[position].annotate(short_name(names[i]), (math.log10(rows[i][0]), rows[i][column] * scale), fontsize=7)
    axes[position].set_title(title)
    axes[position].set_xlabel("log10 of median monthly collection")
    axes[position].set_ylabel(label)
  axes[0].legend(fontsize=8)
  figure.tight_layout()
  save_figure(figure, f"{folder}/cluster_scatter_{tag}.png")
  plt.close(figure)

# region and year mode: shows which cluster each year of a region is in, to see if a region stays in one cluster
def year_consistency(names, labels):
  table = {}
  for name, label in zip(names, labels):
    region, year = name.rsplit(" ", 1)
    table.setdefault(region, {})[int(year)] = label
  rows = []
  print("\n-----------DO THE YEARS OF A REGION STAY IN ONE CLUSTER?-----------")
  print(f"{'Region':<12}" + "".join(f"{year:<7}" for year in range(2020, 2025)) + "Same cluster all years")
  for region, by_year in table.items():
    clusters = [by_year[year] for year in range(2020, 2025)]
    same = len(set(clusters)) == 1
    rows.append([short_name(region)] + clusters + ["yes" if same else "no"])
    print(f"{short_name(region):<12}" + "".join(f"{cluster:<7}" for cluster in clusters) + ("yes" if same else "no"))
  stable = sum(1 for row in rows if row[-1] == "yes")
  print(f"{stable} of {len(rows)} regions stay in one cluster for all five years")
  return rows

# calls the other functions
def kmeans_tax_collection():
  collections = read_collection()
  if collections is None:
    return
  mode = ask_mode()
  tag = "region" if mode == 1 else "region_year"
  if mode == 1:
    names, rows = region_data(collections)
  else:
    names, rows = region_year_data(collections)
  if len(rows) < 4:
    print("Not enough points to cluster.")
    return
  max_k = min(20, len(rows) - 1)
  chosen_k = ask_k(max_k)
  random_gen = random.Random(170)
  for folder in ["tables", "graphs"]:
    os.makedirs(f"results/{folder}", exist_ok=True)

  features = make_features(rows)
  means, stds = column_stats(features)
  points = standardize(features, means, stds)
  print_preprocessed(names, rows, points)
  distances = pairwise_distances(points)

  results = try_all_k(points, distances, random_gen, max_k)
  elbow_k = find_elbow(results)
  best_k, balanced_k = choose_k(results, elbow_k)
  print("\n-----------K SELECTION-----------")
  print(f"{'k':<5}{'Inertia':<14}{'Silhouette':<12}{'Smallest Cluster':<18}")
  for result in results:
    print(f"{result['k']:<5}{result['inertia']:<14.2f}{result['silhouette']:<12.4f}{result['smallest']:<18}")
  print(f"Elbow k: {elbow_k}")
  if [result["smallest"] for result in results if result["k"] == elbow_k][0] < 2:
    print("Note: at the elbow k, one cluster has only 1 point")
  print(f"Best silhouette k (no cluster with 1 point): {best_k}")
  print(f"Balanced k (elbow k if its silhouette is within 0.05 of the best): {balanced_k}")
  if chosen_k == 0:
    chosen_k = balanced_k

  final = [result for result in results if result["k"] == chosen_k][0] # uses the run from the sweep
  print("\n-----------K-MEANS RESULTS-----------")
  print(f"Number of Clusters: {chosen_k}")
  print(f"Inertia: {final['inertia']:.2f}")
  print(f"Inertia over {RESTARTS} restarts: min {min(final['all_inertias']):.2f}, max {max(final['all_inertias']):.2f}")
  print(f"Converged in {final['iterations']} iterations")
  print(f"Mean Silhouette Score: {final['silhouette']:.4f}")
  profiles = cluster_profiles(rows, final, chosen_k)
  display_clusters(names, profiles)
  if mode == 2:
    consistency = year_consistency(names, final["labels"])
    save_table(f"results/tables/year_consistency_{tag}.csv", ["region", "2020", "2021", "2022", "2023", "2024", "same_cluster_all_years"], consistency)

  save_table(f"results/tables/preprocessed_{tag}.csv", ["point", "median_monthly", "median_mom_growth", "median_qoq_growth", "volatility", "z_size", "z_mom_growth", "z_qoq_growth", "z_volatility"],
             [[short_name(names[i])] + rows[i] + points[i] for i in range(len(rows))])
  save_table(f"results/tables/k_selection_{tag}.csv", ["k", "inertia", "silhouette", "smallest_cluster"],
             [[result["k"], result["inertia"], result["silhouette"], result["smallest"]] for result in results])
  save_table(f"results/tables/cluster_profiles_{tag}.csv", ["cluster", "points", "median_monthly", "median_mom_growth", "median_qoq_growth", "volatility", "meaning"],
             [[c, len(profiles[c]["members"]), profiles[c]["median_monthly"], profiles[c]["growth"], profiles[c]["quarter_growth"], profiles[c]["volatility"], profiles[c]["meaning"]] for c in range(chosen_k)])
  save_table(f"results/tables/cluster_assignments_{tag}.csv", ["point", "cluster"],
             [[short_name(names[i]), final["labels"][i]] for i in range(len(rows))])
  plot_k_graphs(results, elbow_k, chosen_k, "results/graphs", tag)
  # figures are turned off for now (remove the # to make them again)
  # plot_profiles(profiles, "results/figures", tag)
  # plot_scatter(names, rows, final["labels"], chosen_k, "results/figures", tag)
  print("\nTables saved inside results/tables, graphs inside results/graphs")

if __name__ == "__main__":
  kmeans_tax_collection()