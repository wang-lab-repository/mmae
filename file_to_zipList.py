import os
import zipfile


def zipDir(dirpath, outFullName):

    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        fpath = path.replace(dirpath, '')

        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


if __name__ == "__main__":
    input_path = "/root/1"
    output_path = "/root/1.zip"

    zipDir(input_path, output_path)
