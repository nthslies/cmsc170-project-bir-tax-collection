# Milestone 2 K-Means Clustering on BIR Tax Collection Data

import math
import random
from filereader import load_clustering

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

# makes the features of each region
def make_features(regions):
  features = []
  for mean_monthly, growth, volatility in regions:
    features.append([math.log10(max(mean_monthly, 1e-9)), growth, volatility]) # log so big regions do not take over
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

# squared distance of two points
def squared_distance(point, centroid):
  total = 0
  for j in range(len(point)):
    total += (point[j] - centroid[j]) ** 2
  return total

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
      new_centroids.append(centroids[c]) # empty cluster, keep the old centroid
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
  centroids = [list(point) for point in random_gen.sample(points, k)]
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
def kmeans(points, k, random_gen, restarts=10):
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

# tries every k from 2 to max_k, one dictionary of results for each k
def try_all_k(points, distances, random_gen, max_k):
  results = []
  for k in range(2, max_k + 1):
    run = kmeans(points, k, random_gen)
    scores = silhouette_scores(distances, run["labels"], k)
    run["k"] = k
    run["silhouette"] = sum(scores) / len(scores)
    results.append(run)
  return results

# calls the other functions
def kmeans_tax_collection():
  regions, region_names = load_clustering()
  if regions is None:
    return
  regions = regions.tolist()
  max_k = min(10, len(regions) - 1)

  print()
  chosen_k = ask_k(max_k)
  print()

  random_gen = random.Random(170)
  features = make_features(regions)
  means, stds = column_stats(features)
  points = standardize(features, means, stds)
  distances = pairwise_distances(points)

  results = try_all_k(points, distances, random_gen, max_k)
  print("K SELECTION")
  print(f"{'k':<5}{'Inertia':<14}{'Silhouette':<12}")
  print("-" * 31)
  for result in results:
    print(f"{result['k']:<5}{result['inertia']:<14.2f}{result['silhouette']:<12.4f}")
  if chosen_k == 0:
    best_silhouette = max(result["silhouette"] for result in results)
    chosen_k = [result["k"] for result in results if result["silhouette"] == best_silhouette][0]

  final = [result for result in results if result["k"] == chosen_k][0]
  print()
  print(f"Number of Clusters: {chosen_k}")
  print(f"Inertia: {final['inertia']:.2f}")
  print(f"Inertia over 10 restarts: min {min(final['all_inertias']):.2f}, max {max(final['all_inertias']):.2f}")
  print(f"Converged in {final['iterations']} iterations")
  print(f"Mean Silhouette Score: {final['silhouette']:.4f}")

kmeans_tax_collection()
