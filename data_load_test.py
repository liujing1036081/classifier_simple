# -*- coding: utf-8 -*-

import xdrlib, sys
import xlrd


def open_excel(file):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception, e:
        print str(e)


# 根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引
# def excel_table_byindex(file, colnameindex=0, by_index=0):
#     data = open_excel(file)
#     table = data.sheets()[by_index]
#     nrows = table.nrows  # 行数
#     ncols = table.ncols  # 列数
#     colnames = table.row_values(colnameindex)  # 某一行数据
#     print(colnames)
#     list = []
#     for rownum in range(1, nrows):
#
#         row = table.row_values(rownum)
#         if row:
#             app = {}
#             for i in range(len(colnames)):
#                 app[colnames[i]] = row[i]
#             list.append(app)
#     return list


# 根据名称获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_name：Sheet1名称

def excel_table_byname(file, colnameindex, by_name):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)
    nrows = table.nrows  # 行数
    colnames = table.row_values(colnameindex)  # 某一行数据
    print(colnames)
    list = []
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        if row:
            app = {}
            for i in range(len(colnames)):
                app[colnames[i]] = row[i]
            list.append(app)
    return list


def main():
    filename = 'test2.xlsx'
    by_name = u'Sheet1'
    tables = excel_table_byname(filename, 0, by_name)
    for row in tables:

        print row

if __name__ == "__main__":
    main()
