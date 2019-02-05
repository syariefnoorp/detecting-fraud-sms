# tm2018-g9

## Development Environment :
```
Code Language : Python version 3.7.1
IDE : PyCharm
Library :
- numpy
- pandas
- sastrawi
- sklearn
```
## File Source Code :
```
1. preprocessing.py
Berisi proses untuk melakukan pemecahan dataset menjadi 20% dari tiap class untuk menjadi 
data testing dan 80% dari tiap class untuk menjadi data training, kemudian pada data training
dilakukan proses preprocessing hingga di dapatkan term untuk dilakukan perhitungan.

2. naive_bayes.py
Hasil term dari file preprocessing.py akan digunakan untuk melakukan perhitungan multinomial
naive bayes dan akan di tes pada data testing untuk mendapatkan akurasi penentuan label sms.

3. data_testing.csv
Berisi hasil pemecahan 20% dari tiap class pada data set untuk dijadikan data testing/data uji.

4. data_training.csv
Berisi hasil pemecahan 80% dari tiap class pada data set untuk dijadikan data training/data latih.

5. dataset_sms_ori.csv
Berisi data set original yang didapatkan dari classroom.

6. training_stemming.csv
Hasil output stemming pada data_training.csv yang dilakukan pada file preprocessing.py.
```