# Utils
# import joblib
import pickle
import sklearn
import pandas as pd
import os
import logger
from CustomPDF import CustomPDF

# Tela
from tkinter import Tk
from tkinter import END, N, W, E, S  # constantes do tkinter
from tkinter import ttk, messagebox, Label, Text
from tkinter import filedialog as dlg
from PIL import ImageTk, Image

# Distancias
from distances import mainDistances as dist
myimg = ""
img_path = ""
result_string = ""
img_path_tail = ""

log = logger.getLogger(__file__)


def predizer(root, lbl_imagem, lbl_status, txt_resultado, btn_relatorio):
    # CASOS = 1
    # CONTROLE = 0

    global myimg
    global img_path
    global result_string
    global img_path_tail

    try:
        log.info("Finding image")
        # Selecionar a imagem desejada localmente
        filetypes = [("all files", "*.*")]
        img_path = dlg.askopenfilename(initialdir=os.path.expanduser("~"),
                                       title="Selecionar imagem", filetypes=filetypes)
    except IOError as ioe:
        img_path = ""
        lbl_status.configure(text="Ocorreu um erro, tente novamente.")
        root.update_idletasks()
        log.info("Não foi possível encontrar a imagem selecionada.")
        log.info(f"Erro: {ioe}")

    if img_path is not "":
        log.info("Image found, processing...")
        lbl_status.configure(text="Aguarde, esta operação pode demorar...")

        # Se for encontrado um endereço válido de imagem, adicionamos ela no programa
        myimg = ImageTk.PhotoImage(Image.open(img_path).resize((400, 300)))
        lbl_imagem.configure(image=myimg)
        lbl_imagem.grid(column=1, row=0, rowspan=4)

        # Limpamos a caixa de texto com algum resultado anterior e bloqueamos o botão de relatório pois não
        # foi gerada um resultado pra imagem recém selecionada
        btn_relatorio.configure(state='disabled')

        txt_resultado.configure(state='normal')
        txt_resultado.delete('1.0', END)
        txt_resultado.configure(state='disabled')

        root.update_idletasks()

        # separamos o path da imagem pra utilizar somente a parte importante
        head, tail = os.path.split(img_path)
        img_path_tail = tail

        log.info("Finding distances and predicting for " + tail)

        # try:
        # Neste trecho de código utilizamos o algoritmo do Henrique para gerar as distâncias
        target = pd.DataFrame.from_dict(dist.face_distances(img_path), orient='index').T
        if os.path.isfile("./models/features_to_delete.csv"):
            features_to_delete = pd.read_csv('./models/features_to_delete.csv')
            features_to_delete = features_to_delete.values.tolist()[0]
            target = target.drop(features_to_delete, axis=1)

        target = target.to_numpy()
        log.info("All distances found")

        # Carregamos o modelo de Machine Learn que foi determinado como o melhor
        # O único problema aqui é o caminho que tem que ser alterado manualmente

        log.info("Loading ML model")

        if os.path.isfile('./models/red_dim.sav'):
            log.info("Loading Dimensionality Reduction")
            red_dim = pickle.load(open("./models/red_dim.sav", 'rb'))
            log.info("Applying Dimensionality Reduction")
            target = red_dim.transform(target)

        model = pickle.load(open("./models/model.sav", 'rb'))
        # model = joblib.load('./models/rf.joblib')

        log.info("Model loaded successfully")

        log.info("Predicting...")

        # Fazemos a predição
        result = model.predict(target)
        if result[0] == 1:
            result_string = "TEA"
        elif result[0] == 0:
            result_string = "Desenvolvimento típico"
        else:
            result_string = "ERROR"

            # Agora vamos atualizar a caixa de texto com o resultado
            # Também liberamos o botão de relatório
        txt_resultado.configure(state='normal')

        txt_resultado.insert('end', 'Imagem: ' + tail+'\n')
        txt_resultado.insert('end', '\n')
        txt_resultado.insert('end', 'Diagnóstico: ' + result_string+'\n')
        lbl_status.configure(text="Operação finalizada!")

        txt_resultado.configure(state='disabled')

        btn_relatorio.configure(state='active')

        log.info("Result for {0}: {1}".format(tail, str(result[0])))
        # except:
        #     lbl_status.configure(text="Ocorreu um erro, tente novamente.")
        #     log.info("Ocorreu um erro, tente novamente.")
        #     root.update_idletasks()

    else:
        messagebox.showerror(title="Erro", message="Nenhuma imagem foi selecionada!")
        raise Exception("Nenhuma imagem foi selecionada!")


def gerarRelatorio():
    # Documentação FPDF: https://pyfpdf.readthedocs.io/en/latest/
    global result_string
    global img_path
    global img_path_tail

    filetypes = [("all files", "*.*")]
    log.info("Choosing path to save")
    path_to_save = dlg.asksaveasfilename(initialdir=os.path.expanduser("~"),
                                         title="Select Path", filetypes=filetypes) + ".pdf"

    if not os.path.isfile(path_to_save):
        log.info("Preparing PDF file")
        pdf = CustomPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        # pdf.ln(5)
        pdf.cell(190, 10, txt='Imagem: ' + img_path_tail, border=1, ln=1)
        pdf.cell(190, 10, txt='Diagnóstico: ' + result_string, border=1, ln=1)  # wide, high, ln=1 (linebreak)
        pdf.ln(5)

        pdf.image(img_path, x=45, w=120)
        pdf.ln(5)
        pdf.image('./output/' + img_path_tail, x=45, w=120)

        log.info("Saving on {0}".format(path_to_save))
        pdf.output(path_to_save)
        log.info("PDF file saved!")
    else:
        messagebox.showerror(title="Erro", message="Não foi possível salvar o relatório com este nome!")
        log.info("It was not possible to save this file")
        raise Exception("Já existe um arquivo com este nome")


def mainScreen():
    log.info("Loading interface")
    root = Tk()
    root.title("Sistema de auxílio ao diagnóstico de transtornos psiquiátricos")
    root.geometry("770x350")  # LarguraxAltura+DistEsq+DistCima

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    root.resizable(0, 0)  # prevent the window from maximizing

    # Setar a frame onde vão ficar os botões para carregar a imagem, predizer e gerar o relatório
    config_frame = ttk.Frame(root, padding="10", relief='solid', borderwidth=1)
    config_frame.grid(column=0, row=0, sticky=(N, W, E, S), padx=10, pady=10)

    # Imagem
    lbl_imagem = ttk.Label(config_frame)

    # Crir os botões e a caixa de texto para o diagnóstico
    txt_resultado = Text(config_frame, height=10, width=37, state='disabled')
    lbl_status = Label(config_frame, text="")
    btn_predict = ttk.Button(config_frame, text='Nova imagem', command=lambda: predizer(root,
                                                                                        lbl_imagem,
                                                                                        lbl_status,
                                                                                        txt_resultado,
                                                                                        btn_relatorio))
    btn_relatorio = ttk.Button(config_frame, text='Salvar PDF', command=gerarRelatorio, state='disabled')

    # Organizar os widgets na tela
    btn_predict.grid(column=0, row=0, sticky=(W, E))
    lbl_status.grid(column=0, row=1, sticky=(W, E))
    btn_relatorio.grid(column=0, row=3, sticky=(W, E))
    txt_resultado.grid(column=0, row=2, sticky=(N, S, W, E))

    # definir o espaçamento entre os widgets
    for child in config_frame.winfo_children():
        child.grid_configure(padx=5, pady=5)

    log.info("Interface loaded without errors!")

    root.bind('<Return>')

    root.mainloop()


if __name__ == '__main__':
    mainScreen()
