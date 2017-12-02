import helpers

num_clusters, original_labels, points = helpers.generate_data_points()

helpers.display(points, original_labels)

(centers, labels, it) = helpers.kmeans(points, num_clusters)
print('Centers found by algorithm: ')
print(centers)
print('Total iterations: {}'.format(it))

helpers.display(points, labels)
