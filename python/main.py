import functions as fun
import warnings

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    LABELS = True

    #Wczytanie danych
    data = fun.read("all_summary.txt",";")
    test_data = fun.read("test_data.txt",",")

    #filtrowanie danych
    data = fun.filter_data(data,["DA","DC","DT","DU","DG", "DI","UNK","UNX","UNL","PR","PD","Y1","EU","N","15P","UQ","PX4", "NAN"])

    if(LABELS):
        labels = fun.read("labels.txt",",")
        data['res_name']=labels['res_name_group']

    #uczenie
    fun.learn(data,test_data)



