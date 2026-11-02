# Milestone 2 K-Means Clustering on BIR Tax Collection Data

import math
from filereader import load_clustering

# asks for the number of clusters
def ask_k(total_regions):
  while True:
    try:
      k = int(input("Enter the number of clusters, k: "))
      if 2 <= k <= total_regions:
        return k
      print(f"k must be from 2 to {total_regions}!")
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

# puts every region in the cluster of its nearest centroid
def assign_clusters(points, centroids):
  labels = []
  for point in points:
    best = 0
    best_distance = squared_distance(point, centroids[0])
    for c in range(1, len(centroids)):
      distance = squared_distance(point, centroids[c])
      if distance < best_distance: # strictly smaller, so ties go to the first centroid
        best = c
        best_distance = distance
    labels.append(best)
  return labels

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

# number of regions in each cluster
def cluster_sizes(labels, k):
  sizes = [0] * k
  for label in labels:
    sizes[label] += 1
  return sizes

# runs k-means until the clusters stop changing
def kmeans(points, k):
  print("Initializing centroids... ")
  centroids = [list(point) for point in points[:k]] # first k regions
  labels = None
  iteration = 0
  while True:
    iteration += 1
    new_labels = assign_clusters(points, centroids)
    print(f"updating centroid [iteration]: {iteration}")
    if new_labels == labels: # nothing moved, so the centroids won't change anymore
      break
    labels = new_labels
    centroids = update_centroids(points, labels, centroids)
  return labels, centroids

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

# prints the number of regions and the regions of each cluster
def display_clusters(labels, region_names, k):
  sizes = cluster_sizes(labels, k)
  print("Regions per Cluster")
  print(f"{'Cluster':<11}{'# Regions':<11}{'Regions':<80}")
  print("-" * 102)
  for c in range(k):
    members = ", ".join(region_names[i] for i in range(len(labels)) if labels[i] == c)
    print(f"{c:<11}{sizes[c]:<11}{members}")

# calls the other functions
def kmeans_tax_collection():
  regions, region_names = load_clustering()
  if regions is None:
    return
  regions = regions.tolist()

  print()
  k = ask_k(len(regions))
  print()

  features = make_features(regions)
  means, stds = column_stats(features)
  points = standardize(features, means, stds)
  print(f"Total # of regions: {len(regions)}")
  print("Features: log10 of mean monthly collection, average annual growth, volatility (standardized)")
  print()
  print("################################################")
  print()
  print(f"Finding k={k} clusters...")
  print()
  labels, centroids = kmeans(points, k)
  print()
  display_clusters(labels, region_names, k)
  print()
  distances = pairwise_distances(points)
  scores = silhouette_scores(distances, labels, k)
  print(f"Overall Silhouette score: {sum(scores) / len(scores)}")

kmeans_tax_collection()