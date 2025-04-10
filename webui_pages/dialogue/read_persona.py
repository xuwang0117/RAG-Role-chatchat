import openpyxl

# 读取Excel文件
def read_persona(file_path):
    # 打开文件
    workbook = openpyxl.load_workbook(file_path)
    # 选择活动工作表
    sheet = workbook.active
    # 存储结果
    result = []
    personas_name ,personas_description, personas_resume = [], [], []

    # 遍历每一行（从第一行开始）
    for row in sheet.iter_rows(min_row=2, max_col=3):
        # 获取每一行的第一列和第二列
        col1 = row[0].value
        col2 = row[1].value
        col3 = row[2].value
        # 拼接字符串，格式为 "第一列:第二列"
        if col1 and col2:  # 检查两列是否有值
            result.append(f"{col1}：{col2}")
            personas_name.append(col1)
            personas_description.append(col2)
        if col3:
            personas_resume.append(col3)
        else:
            personas_resume.append(" ")
    
    # 返回结果，使用换行符隔开每行内容
    return "\n".join(result), personas_name,personas_description, personas_resume

# # 假设文件路径为 'example.xlsx'
# file_path = 'persona.xlsx'
# output = read_persona(file_path)
# print(output)
