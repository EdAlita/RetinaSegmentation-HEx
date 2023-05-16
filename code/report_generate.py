import os
# from pyhtml2pdf import converter
import numpy as np

def reportGen(info,img_path,data,data1):
    img_list =img_path[0]
    img_list=os.listdir(img_list)
    precision_p=np.sum(data)/len(img_list)
    recall_p=np.sum(data1)/len(img_list)
    html_string='<html><head> <style>table, th, td {border: 1px solid black;border-collapse: collapse;}</style><title></title></head><body>'
    html_string +="<h1>HardExudate Report</h1>"
    html_string +="<h2>Precision: "+  str(round(precision_p,3)) +"</h2>"
    html_string +="<h2>Recall: "+  str(round(recall_p,3)) +"</h2>"
    html_string +="<table><tr>"
    for head in range(0,len(info)):
        html_string +="<th width='40%'>"+ info[head] +"</th>"
    html_string +="</tr>"
   
    for sl in range(0, len(img_list)):
        html_string +="<tr>"
        for path in range(0, len(img_path)):
            ld=os.listdir(img_path[path])
            html_string +="<td>"+ ld[sl] +"<br/><img width='350' src='"+ img_path[path] + ld[sl] +"'></td>"

        html_string +="<td>"+ str(data[sl]) +"</td>"
        html_string +="<td>"+ str(data1[sl]) +"</td>"
        html_string +="</tr>"

    html_string +='</table></body></html>'
    f = open('Results/report.html','w')
    f.write(html_string)
    f.close()

    # pdf report Generator
    # path = os.path.abspath('Results/report.html')
    # converter.convert(f'file:///{path}', 'Results/report.pdf')
