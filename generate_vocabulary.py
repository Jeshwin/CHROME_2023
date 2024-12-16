import os
import re

directory = "vocabulary"

with open("vocabulary.txt", "x") as output_file:
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            with open(os.path.join(directory, filename), "r") as html_file:
                html_content = html_file.read()
                latex_content = re.findall(
                    r'<annotation encoding="application/x-tex">(.*?)</annotation>',
                    html_content,
                    re.DOTALL,
                )

                for tex_expr in latex_content:
                    tex_expr = tex_expr.replace("&lt;", "<")
                    tex_expr = tex_expr.replace("&gt;", ">")
                    tex_expr = tex_expr.replace("&amp;", "&")
                    tex_expr = tex_expr.strip() + "\n"
                    print(tex_expr)
                    output_file.write(tex_expr.strip() + "\n")
