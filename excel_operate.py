import xlrd
# todo 对excel的操作

# todo 打开excle
xl = xlrd.open_workbook('D:\数学建模美赛\运算数据\运动员.xlsx')
# print(xl.read())

# todo 通过索引获取工作表
table = xl.sheets()[0]
print(table)

# 获取一共多少行
rows = table.nrows
print(rows)

# todo 获取第一行的内容,索引从0开始
row = table.row_values(0)
print(row)

# todo 获取第一列的整列的内容
col = table.col_values(0)
print(col)

# todo 获取单元格值，第几行第几个，索引从0开始
data = table.cell(1, 0).value
print(data)
