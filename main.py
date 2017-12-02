import helpers
from sklearn.cluster import KMeans

num_clusters, original_labels, points = helpers.generate_data_points()

helpers.display(points, original_labels)

(centers, labels, it) = helpers.kmeans(points, num_clusters)
print('Centers found by algorithm: ')
print(centers)
print('Total iterations: {}'.format(it))

helpers.display(points, labels)

kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
print('Centers found by scikit-learn library: ')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(points)
print('Total iterations using scikit-learn library: {}'.format(kmeans.n_iter_))

helpers.display(points, pred_label)