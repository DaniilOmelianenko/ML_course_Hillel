import numpy as np
from numpy import ndarray, int64


def mount_google_drive() -> bool | None:
    """
    Function to mount Google Drive.
    :return: True if mounting is successful, None otherwise.
    """
    try:
        from google.colab.drive import mount

        mount(mountpoint="/content/drive")
        return True

    except Exception as error:
        print(f"Error while mounting Google Drive: {error}")


def get_google_drive_dataset_path() -> str:
    """
    Function to retrieve the path of the raw dataset from Google Drive.
    :return: Path of raw dataset as a string.
    """
    google_drive_dataset_path: str = "/content/drive/MyDrive/Hillel/Machine_Learning_Course/HW2/Datasets/iris.data"
    return google_drive_dataset_path


def get_dataset_source_url() -> str:
    """
    Function to retrieve the URL of the source dataset.
    :return: URL of the source dataset as a string.
    """
    source_dataset_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    return source_dataset_url


def get_numpy_array(dataset_path: str) -> np.ndarray | None:
    """
    Function to convert the dataset into a NumPy array.
    :param dataset_path: Path or URL of the dataset.
    :return: NumPy array containing the dataset, or None if errors occur.
    """
    try:
        return np.genfromtxt(fname=dataset_path, delimiter=',')

    except Exception as error:
        print(f"Error while converting dataset to NumPy array: {error}")


def main() -> ndarray | None:
    """
    Main function to start the app.
    :return: NumPy array containing the dataset, or None if errors occur.
    """
    if mount_google_drive():
        import os

        google_drive_dataset_path: str = get_google_drive_dataset_path()
        if os.path.exists(path=google_drive_dataset_path):
            dataset_path: str = google_drive_dataset_path

        else:
            dataset_path: str = get_dataset_source_url()

    else:
        dataset_path: str = get_dataset_source_url()

    return get_numpy_array(dataset_path=dataset_path)


# if __name__ == "__main__":
#     raw_array: ndarray | None = main()
#     print((raw_array))

# arr1 = np.array([1, 2, 3, 4, 5], ndmin=1)
# print(arr1, arr1.ndim)
#
# arr2 = np.array(arr1, ndmin=1)
# print(arr1 is arr2)
# print(arr2, arr2.ndim)
# arr1 += 2
# print(arr1, arr1.ndim)
# print(arr2, arr2.ndim)
#
# if clean_array.ndim != 2:
#     clean_2d_array: ndarray = np.array(object=clean_array, ndmin=2)
#
# else:
#     clean_2d_array: ndarray = np.copy(a=clean_array)
#
# print(f"This is {clean_2d_array.ndim}D array: \n{clean_2d_array}")

data = np.array([[5.1, 3.5, 1.4, 0.2],
                 [4.9, 3., 1.4, 0.2],
                 [4.7, 3.2, 1.3, 0.2],
                 [4.6, 3.1, 1.5, 0.2],
                 [5., 3.6, 1.4, 0.2],
                 [5.4, 3.9, 1.7, 0.4],
                 [4.6, 3.4, 1.4, 0.3],
                 [5., 3.4, 1.5, 0.2],
                 [4.4, 2.9, 1.4, 0.2],
                 [4.9, 3.1, 1.5, 0.1],
                 [5.4, 3.7, 1.5, 0.2],
                 [4.8, 3.4, 1.6, 0.2],
                 [4.8, 3., 1.4, 0.1],
                 [4.3, 3., 1.1, 0.1],
                 [5.8, 4., 1.2, 0.2],
                 [5.7, 4.4, 1.5, 0.4],
                 [5.4, 3.9, 1.3, 0.4],
                 [5.1, 3.5, 1.4, 0.3],
                 [5.7, 3.8, 1.7, 0.3],
                 [5.1, 3.8, 1.5, 0.3],
                 [5.4, 3.4, 1.7, 0.2],
                 [5.1, 3.7, 1.5, 0.4],
                 [4.6, 3.6, 1., 0.2],
                 [5.1, 3.3, 1.7, 0.5],
                 [4.8, 3.4, 1.9, 0.2],
                 [5., 3., 1.6, 0.2],
                 [5., 3.4, 1.6, 0.4],
                 [5.2, 3.5, 1.5, 0.2],
                 [5.2, 3.4, 1.4, 0.2],
                 [4.7, 3.2, 1.6, 0.2],
                 [4.8, 3.1, 1.6, 0.2],
                 [5.4, 3.4, 1.5, 0.4],
                 [5.2, 4.1, 1.5, 0.1],
                 [5.5, 4.2, 1.4, 0.2],
                 [4.9, 3.1, 1.5, 0.1],
                 [5., 3.2, 1.2, 0.2],
                 [5.5, 3.5, 1.3, 0.2],
                 [4.9, 3.1, 1.5, 0.1],
                 [4.4, 3., 1.3, 0.2],
                 [5.1, 3.4, 1.5, 0.2],
                 [5., 3.5, 1.3, 0.3],
                 [4.5, 2.3, 1.3, 0.3],
                 [4.4, 3.2, 1.3, 0.2],
                 [5., 3.5, 1.6, 0.6],
                 [5.1, 3.8, 1.9, 0.4],
                 [4.8, 3., 1.4, 0.3],
                 [5.1, 3.8, 1.6, 0.2],
                 [4.6, 3.2, 1.4, 0.2],
                 [5.3, 3.7, 1.5, 0.2],
                 [5., 3.3, 1.4, 0.2],
                 [7., 3.2, 4.7, 1.4],
                 [6.4, 3.2, 4.5, 1.5],
                 [6.9, 3.1, 4.9, 1.5],
                 [5.5, 2.3, 4., 1.3],
                 [6.5, 2.8, 4.6, 1.5],
                 [5.7, 2.8, 4.5, 1.3],
                 [6.3, 3.3, 4.7, 1.6],
                 [4.9, 2.4, 3.3, 1.],
                 [6.6, 2.9, 4.6, 1.3],
                 [5.2, 2.7, 3.9, 1.4],
                 [5., 2., 3.5, 1.],
                 [5.9, 3., 4.2, 1.5],
                 [6., 2.2, 4., 1.],
                 [6.1, 2.9, 4.7, 1.4],
                 [5.6, 2.9, 3.6, 1.3],
                 [6.7, 3.1, 4.4, 1.4],
                 [5.6, 3., 4.5, 1.5],
                 [5.8, 2.7, 4.1, 1.],
                 [6.2, 2.2, 4.5, 1.5],
                 [5.6, 2.5, 3.9, 1.1],
                 [5.9, 3.2, 4.8, 1.8],
                 [6.1, 2.8, 4., 1.3],
                 [6.3, 2.5, 4.9, 1.5],
                 [6.1, 2.8, 4.7, 1.2],
                 [6.4, 2.9, 4.3, 1.3],
                 [6.6, 3., 4.4, 1.4],
                 [6.8, 2.8, 4.8, 1.4],
                 [6.7, 3., 5., 1.7],
                 [6., 2.9, 4.5, 1.5],
                 [5.7, 2.6, 3.5, 1.],
                 [5.5, 2.4, 3.8, 1.1],
                 [5.5, 2.4, 3.7, 1.],
                 [5.8, 2.7, 3.9, 1.2],
                 [6., 2.7, 5.1, 1.6],
                 [5.4, 3., 4.5, 1.5],
                 [6., 3.4, 4.5, 1.6],
                 [6.7, 3.1, 4.7, 1.5],
                 [6.3, 2.3, 4.4, 1.3],
                 [5.6, 3., 4.1, 1.3],
                 [5.5, 2.5, 4., 1.3],
                 [5.5, 2.6, 4.4, 1.2],
                 [6.1, 3., 4.6, 1.4],
                 [5.8, 2.6, 4., 1.2],
                 [5., 2.3, 3.3, 1.],
                 [5.6, 2.7, 4.2, 1.3],
                 [5.7, 3., 4.2, 1.2],
                 [5.7, 2.9, 4.2, 1.3],
                 [6.2, 2.9, 4.3, 1.3],
                 [5.1, 2.5, 3., 1.1],
                 [5.7, 2.8, 4.1, 1.3],
                 [6.3, 3.3, 6., 2.5],
                 [5.8, 2.7, 5.1, 1.9],
                 [7.1, 3., 5.9, 2.1],
                 [6.3, 2.9, 5.6, 1.8],
                 [6.5, 3., 5.8, 2.2],
                 [7.6, 3., 6.6, 2.1],
                 [4.9, 2.5, 4.5, 1.7],
                 [7.3, 2.9, 6.3, 1.8],
                 [6.7, 2.5, 5.8, 1.8],
                 [7.2, 3.6, 6.1, 2.5],
                 [6.5, 3.2, 5.1, 2.],
                 [6.4, 2.7, 5.3, 1.9],
                 [6.8, 3., 5.5, 2.1],
                 [5.7, 2.5, 5., 2.],
                 [5.8, 2.8, 5.1, 2.4],
                 [6.4, 3.2, 5.3, 2.3],
                 [6.5, 3., 5.5, 1.8],
                 [7.7, 3.8, 6.7, 2.2],
                 [7.7, 2.6, 6.9, 2.3],
                 [6., 2.2, 5., 1.5],
                 [6.9, 3.2, 5.7, 2.3],
                 [5.6, 2.8, 4.9, 2.],
                 [7.7, 2.8, 6.7, 2.],
                 [6.3, 2.7, 4.9, 1.8],
                 [6.7, 3.3, 5.7, 2.1],
                 [7.2, 3.2, 6., 1.8],
                 [6.2, 2.8, 4.8, 1.8],
                 [6.1, 3., 4.9, 1.8],
                 [6.4, 2.8, 5.6, 2.1],
                 [7.2, 3., 5.8, 1.6],
                 [7.4, 2.8, 6.1, 1.9],
                 [7.9, 3.8, 6.4, 2.],
                 [6.4, 2.8, 5.6, 2.2],
                 [6.3, 2.8, 5.1, 1.5],
                 [6.1, 2.6, 5.6, 1.4],
                 [7.7, 3., 6.1, 2.3],
                 [6.3, 3.4, 5.6, 2.4],
                 [6.4, 3.1, 5.5, 1.8],
                 [6., 3., 4.8, 1.8],
                 [6.9, 3.1, 5.4, 2.1],
                 [6.7, 3.1, 5.6, 2.4],
                 [6.9, 3.1, 5.1, 2.3],
                 [5.8, 2.7, 5.1, 1.9],
                 [6.8, 3.2, 5.9, 2.3],
                 [6.7, 3.3, 5.7, 2.5],
                 [6.7, 3., 5.2, 2.3],
                 [6.3, 2.5, 5., 1.9],
                 [6.5, 3., 5.2, 2.],
                 [6.2, 3.4, 5.4, 2.3],
                 [5.9, 3., 5.1, 1.8]])

np.random.seed(seed=1729)

# data_shape: tuple = data.shape
# total_elements: int = data.size
#
# random_indices: ndarray = np.random.choice(a=total_elements, size=20, replace=False)
#
# # Способ №1
# array_1: ndarray = data.copy()
# array_1.ravel()[random_indices] = np.nan
#
# # Способ №1
# array_2: ndarray = data.copy()
# row_indices, col_indices = np.unravel_index(indices=random_indices, shape=array_2.shape)
# array_2[row_indices, col_indices] = np.nan
#
#
# # print(data)
# print(array_1)
# print(array_2)
# print(sum(np.isnan(array_2)))
# print(np.count_nonzero(np.isnan(array_2)))
# print(array_2.ndim, array_1.ndim)

arr = np.array([[6.0, 2.0, 3.0, 5.6],
                [3.0, 5.0, 6.0, 9.7],
                [4.5, 7.0, 6.0, 3.3],
                [2.0, 5.0, 6.0, 3.3],
                [7.0, 8.0, 1.0, 2.5]])

array_filter: ndarray = (arr[:, 2] > 1.5) & (arr[:, 0] < 5.0)

# Фильтрация массива и сохранение в другую переменную
filtered_arr = arr[array_filter].copy()

# print(top_array)
# print(bottom_array)
arr = np.array([[3, 2, 4],
                [1, 6, 2],
                [5, 8, 7],
                [2, 8, 7],
                [9, 81, 17],
                [0, 77, 7],
                [0, 8, 7],
                [0, 9, 1]])

# Сортировка массива по первой колонке в порядке возрастания
sorted_arr1: ndarray = arr[np.argsort(a=arr[:, 0])]
sorted_arr2 = arr[np.argsort(a=-arr[:, 0])]

# Вывод результатов
# print(sorted_arr1)
# print(sorted_arr2)
united_array: ndarray = np.vstack([sorted_arr1, sorted_arr2])


# print(united_array)


def print_array_properties(array: ndarray, title: str | None = None) -> None:
    """
    Accepts an array as input and prints its properties.
    :param array: NumPy array.
    :param title: Array title.
    """
    if title:
        print(f"{title} :")
    print(array)
    print(f"Array  ndim: {array.ndim}")
    print(f"Array shape: {array.shape}")
    print(f"Array dtype: {array.dtype}")
    print(f"Array  size: {array.size} elements")
    print(f"Array  type: {type(array)}")
    print()


arr = np.array([[3, 2, 4],
                [1, 6, 2],
                [5, 8, 3],
                [2, 8, 7],
                [9, 81, 16],
                [0, 77, 12],
                [0, 8, 6.5],
                [0, 9, 1]])


def modify_array_column(array: ndarray,
                        column_index: int = 0,
                        column_multiplier: int = 2,
                        column_divider: int = 4) -> ndarray:
    """
    The function accepts a NumPy array and the index of a specific column.
    If the values in the column are less than its mean(),
    they will be multiplied by the "column_multiplier" value.
    Otherwise, they will be divided by the "column_divider" value.
    :param array: An NumPy array.
    :param column_index: Index of the column in the array.
    :param column_multiplier: Value by which to multiply the column values.
    :param column_divider: Value by which to divide the column values.
    :return: Array with the modified column.
    """
    column: ndarray = array[:, column_index]
    column_mean: np.float64 = np.mean(column)
    print(f"Mean: {column_mean}")
    print(f"Old column: {column}")

    modified_column: ndarray = np.where(column < column_mean, column * column_multiplier, column / column_divider)
    print(f"New column: {modified_column}")
    array[:, column_index] = modified_column
    return array


#
# print(F"New array:\n{modify_array_column(array=arr.copy(), column_index=2)}")
# print(f"OLD array:\n{arr}")
#
# array_with_modified_column_f32: ndarray = arr.astype(dtype=np.float32)
#
# print(arr.reshape(4, 6))
def get_possible_shapes(number_of_elements: int) -> ndarray:
    """
    Function to calculate all possible shapes of array with "number_of_elements" size.
    :param number_of_elements: Size of array.
    :return: NumPy array contents all possible shapes.
    """
    divisors: ndarray = np.arange(start=1, stop=int(np.sqrt(number_of_elements)) + 1)
    mask: ndarray = number_of_elements % divisors == 0
    divisors: ndarray = divisors[mask]
    shapes: ndarray = np.column_stack(tup=(divisors, number_of_elements // divisors))

    reversed_array = shapes[::-1]
    swapped_array = reversed_array[:, ::-1]
    shapes: ndarray = np.vstack(tup=[shapes, swapped_array])

    return shapes


shapes: ndarray = get_possible_shapes(number_of_elements=600)
# print(shapes)
arr = np.array([[3, 2, 4],
                [1, 6, 2],
                [5, 8, 3],
                [2, 8, 7],
                [9, 81, 16],
                [0, 77, 12],
                [0, 8, 6.5],
                [0, 9, 1]])
print(arr[0:2, 0:2])

import numpy as np
import matplotlib.pyplot as plt

def show_image(image: ndarray, title: str):
  plt.imshow(X=image)
  plt.title(label='Original Image')
  plt.show()

gray_python = np.dot(a=TEST_IMAGE[..., :3], b=[100, 100, 100])

TEST_IMAGE: ndarray = plt.imread(fname="/content/drive/MyDrive/Hillel/Machine_Learning_Course/HW2/Datasets/python.jpeg")