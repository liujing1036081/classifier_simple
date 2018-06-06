#! -*- coding: utf-8 -*-

import xlrd

'''
读取excel文件
获取训练集文本和类别标签'''

# 打开excel文件
def open_excel(file):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception, e:
        print str(e)

# 获取数据
def excel_table_byname(file, colnameindex, by_name):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)  # 根据名字读取excel的某一个数据表
    nrows = table.nrows  # 行数
    colnames = table.row_values(colnameindex)  # 某一行数据值的列表
    text_list = []
    label_list = []
    for rownum in range(1, nrows):
        row = table.row_values(rownum)
        if row:
            # for i in range(len(colnames)):
            if colnames[1] == u'问句' and row[5] != '':
                text_list.append(row[1])
                if row[5] == u'对主播的印象':
                    label_list.append(int(0))
                if row[5] == u'用户对自己的描述':
                    label_list.append(int(1))
                if row[5] == u'直播相关问题':
                    label_list.append(int(2))

    return text_list, label_list