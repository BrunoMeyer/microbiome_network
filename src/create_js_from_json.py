import os


def create_js_from_json():
    js_file_content = "var database_service = {\n"
    for filename in os.listdir('json'):
        if filename.split(".")[-1] == "json":
            with open("json/{}".format(filename)) as json_file:
                json_content = json_file.read()
                js_file_content+="\t\"{}\": {},\n".format(filename,json_content)
            
                json_file.close()

    js_file_content += "};"

    with open('js/database_service.js', 'w') as file:
        file.write(js_file_content)
        file.close()