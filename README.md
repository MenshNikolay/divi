<h1>Данный проект создан с использованием интилектуальной собственности компании 3DiVi</h1>
<h2>Описание</h2>
<p>Выполнение программы происходит в файле main.py, для пакетной обрабоки данных нужно указать абсолютный путь к директории с расположением датасета, в формате string. Фунция main принимает в качетсве аргумента путь к датасету и количество экземпляров из датасета, если
не требуется полная обработка датасета.
Для выполнения основной программы требуется 4 модуля. file_handler.py, sri.py, write_csv.py, qaa.py.
1) file_handler.py. Данный модуль производит поиск данных для обработки. Принимаются форматы все файлы с расширениями ".png",".bmp",".tif",".tiff",".jpg",".jpeg",".ppm"
2)sri.py. Данный модуль производит рандомную выборку файлов если в функции main указан параметр
3)write_csv.py. Данный модуль нужен для записи данных после обработки файлов.
4)qaa.py Данный модуль нужен для оценки качества картинки. В основе модуля лежит код взятый с  https://github.com/3DiVi/face-sdk/blob/master/examples/python/processing_blocks/face_detector.py компании 3DiVi. 
</p>