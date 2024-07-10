from osgeo import gdal
import pandas as pd
import glob
import os
import numpy as np
import cvxpy as cp


def get_file_name(file, suf):
    folder_path = file
    file_extension = f'*.{suf}'
    file_list = glob.glob(os.path.join(folder_path, file_extension))
    path_list = []
    for file_path in file_list:
        path_list.append(file_path)

    return path_list


def tif_to_dict(filename, fid_data_dict):
    gdal.UseExceptions()
    inputPathfile = filename
    ds = gdal.Open(inputPathfile)
    bands_num = ds.RasterCount

    for i in range(1, bands_num + 1):
        print("开始读取第{}波段".format(i))
        band = ds.GetRasterBand(i)
        bandarray = band.ReadAsArray()
        bandarray = np.nan_to_num(bandarray, nan=0)

        for row in range(bandarray.shape[0]):
            for col in range(bandarray.shape[1]):
                key = (row, col)
                value = bandarray[row, col]
                if key in fid_data_dict:
                    fid_data_dict[key].append(value)
                else:
                    fid_data_dict[key] = [value]

    return fid_data_dict


def run(name):
    file_add = name
    ndvi = get_file_name(file_add, 'tif')
    data_dict = {}
    for data in ndvi:
        data_dict = tif_to_dict(data, data_dict)
    return data_dict


def read_tif01(filepath):
    dataset = gdal.Open(filepath)
    col = dataset.RasterXSize  # 图像长度
    row = dataset.RasterYSize  # 图像宽度
    geotrans = dataset.GetGeoTransform()  # 读取仿射变换
    proj = dataset.GetProjection()  # 读取投影
    data = dataset.ReadAsArray()  # 转为numpy格式
    data = data.astype(np.float32)  # 转为float类型
    a = data[0][0]
    data[data == a] = np.nan  # 原因：读取某一个行政区的影像图的时候，往往第一行的第一列值为空值
    return [col, row, geotrans, proj, data]


def save_tif(data, file, output):
    ds = gdal.Open(file)
    shape = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output, shape[1], shape[0], 1, gdal.GDT_Float32)  # 以float类型进行存储
    dataset.SetGeoTransform(ds.GetGeoTransform())
    dataset.SetProjection(ds.GetProjection())
    dataset.GetRasterBand(1).WriteArray(data)


if __name__ == "__main__":

    real_ndvi_data_dict = run('real_ndvi')
    dct_ndvi_data_dict = run('dct_ndvi')
    estarfm_ndvi_data_dict = run('estarfm_ndvi')

    dct = {}
    for (row, col), values_dct in dct_ndvi_data_dict.items():
        if (row, col) in real_ndvi_data_dict:
            values_real = real_ndvi_data_dict[(row, col)]
            dct[(row, col)] = [dct_value - real_value for dct_value, real_value in zip(values_dct, values_real)]

    estarfm = {}
    for (row, col), values_estarfm in estarfm_ndvi_data_dict.items():
        if (row, col) in real_ndvi_data_dict:
            values_real = real_ndvi_data_dict[(row, col)]
            estarfm[(row, col)] = [estarfm_value - real_value for estarfm_value, real_value in
                                  zip(values_estarfm, values_real)]


    matrix = {}
    dct_value = {}
    estarfm_value = {}

    for (row, col) in dct.keys():
        matrix[(row, col)] = np.array([dct[(row, col)], estarfm[(row, col)]])
        matrix_tp = matrix[(row, col)]
        T_matrix = np.transpose(matrix_tp)
        Er = matrix_tp @ T_matrix
        # print(Er)

        n = 2
        Q = Er

        A = np.ones((1, n))
        b = np.array([1.0])

        x = cp.Variable(n)
        constraints = [A @ x == b, x >= 0]
        objective = cp.Minimize(cp.quad_form(x, Q))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        dct_value[(row, col)] = [x.value[0]]
        estarfm_value[(row, col)] = [x.value[1]]


    filename = get_file_name('dct_ndvi', 'tif')
    for file in filename:
        col, row, geotrans, proj, data = read_tif01(file)
        data = np.nan_to_num(data, nan=0)
        for (row, col), value in dct_value.items():
            value_p = value[0]
            data[row, col] = data[row, col] * value_p
        save_tif(data, file, f'{file}_W.tif')

    filename = get_file_name('estarfm_ndvi', 'tif')
    for file in filename:
        col, row, geotrans, proj, data = read_tif01(file)
        data = np.nan_to_num(data, nan=0)
        for (row, col), value in estarfm_value.items():
            value_p = value[0]
            data[row, col] = data[row, col] * value_p
        save_tif(data, file, f'{file}_W.tif')



