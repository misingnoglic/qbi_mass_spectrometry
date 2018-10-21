# Code to serve website

from scripts.biLSTM import BiLSTM, MultiheadAttention
from scripts.models import TestModel
from scripts.trainer import Trainer
import numpy as np
from flask import Flask, render_template, request, jsonify
from parse_data import amino_acid_name_parse
from calculate_frag_mz import get_frag_mz
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

class Config:
    lr = 0.0001
    batch_size = 8
    max_epoch = 50
    gpu = False
    

neutral_loss_choices = [0, 17, 18, 35, 36, 44, 46]
n_neutral_losses = len(neutral_loss_choices)
n_charges = 7    

@app.route('/')
def home():
    return render_template("index_fancy.html")


@app.route('/graph')
def graph():
    return render_template("html_graph_test.html")



@app.route('/output_json')
def output_json():
    protein_name = request.args.get('protein_name')
    has_beginning_charge, name_one_hot_encoded, charge = amino_acid_name_parse(protein_name)
    mz_data, intensities, ion_types, positions = get_prediction(has_beginning_charge, name_one_hot_encoded, charge)
    positions = [int(x) for x in positions]
    image = generate_chart(mz_data, intensities, ion_types, positions)
    d = {
        'protein_name': protein_name,
        'mz_data': mz_data,
        'intensities': intensities,
        'ion_types': ion_types,
        'positions': positions,
        'b64_image': image,
    }
    #image = generate_chart(mz_data, intensities, ion_types, positions)
    return jsonify(d)

def generate_chart(mz_data, intensities, ion_types, positions):
    fig, ax = plt.subplots(figsize=(16, 6),
                           ncols=1,
                           nrows=2,
                           gridspec_kw={"height_ratios": [1, 0.25]})

    ax[0].stem(mz_data, intensities, markerfmt=" ")
    for i in range(len(mz_data)):
        if ion_types[i] == 'y':
            label = '$y_{%s}$' % (str(positions[i]))
        else:
            label = '$b_{%s}$' % (str(positions[i]))
        ax[0].text(x=mz_data[i], y=intensities[i] + 200, s=label, fontsize=10)
    ax[0].set_xlabel('m/Z')
    ax[0].set_ylabel('Intensity')

    plt.tight_layout()
    image = io.BytesIO()
    plt.savefig(image, format='png')
    string =  base64.b64encode(image.getvalue()).decode("utf-8")
    return "data:image/png;base64, "+string


def get_prediction(has_beginning_charge, name_one_hot_encoded, charge):

    opt=Config()
    net = TestModel(input_dim=24,
                    n_tasks=2*n_charges,
                    embedding_dim=256,
                    hidden_dim_lstm=128,
                    hidden_dim_attention=32,
                    n_lstm_layers=2,
                    n_attention_heads=8,
                    gpu=opt.gpu)
    trainer = Trainer(net, opt)
    
    trainer.load('./saved_models/model_flipped_bkp.pth')
    total_intensities = np.random.normal(70000, 20000)  
    X = np.array(name_one_hot_encoded)
    inputs = [(X, 
               (has_beginning_charge*1, charge), 
               np.zeros((X.shape[0]-1, 2*n_charges)))]
    pred = trainer.predict(inputs)[0]
    
    mz_data = []
    intensities = []
    ion_types = []
    positions = []
    peaks = np.where(pred > 0.005)
    for position, peak_type in zip(*peaks):
        b_y = (peak_type >= 7) * 1
        charge = peak_type - b_y * 7 + 1
        if b_y:
          ion_type = 'y'
          pos = (pred.shape[0] + 1) - position - 1
        else:
          ion_type = 'b'
          pos = position + 1
        mz_data.append(get_frag_mz(name_one_hot_encoded, pos, ion_type, charge))
        intensities.append(pred[position, peak_type] * total_intensities)
        ion_types.append(ion_type)
        positions.append(position)

    # Sort the data now
    zipped_data = list(zip(mz_data, intensities, ion_types, positions))
    zipped_data.sort(key=lambda x: x[0])
    mz_data, intensities, ion_types, positions = zip(*zipped_data)
    return mz_data, intensities, ion_types, positions

def get_prediction_fake_label(a1, a2, a3):
  mzs = [ 115.087,  158.093,  159.077,  173.129,  185.165,  213.161,
        215.115,  229.118,  232.141,  247.145,  271.171,  272.161,
        286.152,  300.194,  315.208,  318.182,  325.188,  334.672,
        343.186,  361.184,  371.23 ,  387.204,  388.207,  391.236,
        399.729,  405.214,  434.141,  435.136,  443.227,  455.272,
        456.27 ,  457.204,  460.253,  463.13 ,  478.215,  500.289,
        501.295,  518.299,  570.305,  571.285,  650.326,  667.354,
        668.338,  685.364,  724.41 ,  742.43 ,  781.428,  798.449,
        867.469,  868.463,  885.481,  938.506,  939.502,  956.518,
       1086.574, 1103.587, 1145.626]
  intensities = [   47.1,    20.8,   132.5,   149.5,    51.3,    64.1,    73.3,
          42.4,   424.6,  2833.3,   130.3,   110.5,   161.3,   127.2,
         232.7,   660.8,   210.8,    57.8,   862.4,   251.8,   171. ,
         709.8,   139.7,   152.3,   119.7,   332.9,    56.2,   105.2,
          82.3,   121.5,    37.1,    35.5,   332.6,   107.4,    21. ,
         204. ,    72.2,   119.6,    21.2,    62.6,    52.1,    63.7,
         568.8, 10000. ,    44. ,   170. ,    62.5,  1017.4,    56.5,
          55.3,  2614.4,   196.6,   111.6,  5127.3,    21. ,   517.8,
          21.4]
  ion_types = [0 for x in intensities]
  positions = [0 for x in intensities]
  return mzs, intensities, ion_types, positions

def get_prediction_fake_prediction(a1, a2, a3):
  mzs = [203.11026147425002,
 232.14041589464,
 247.14410429275,
 318.18121807746,
 343.18502035825003,
 361.18300898261,
 399.727052346815,
 405.21324648173004,
 460.2514228956,
 518.29731045886,
 685.36276424973,
 798.44682822686,
 885.47885663113,
 956.51597041584,
 1103.58438432883]
  intensities = [535.0086225224659,
 716.148171511665,
 3098.8953090921045,
 1107.2814079657198,
 922.5066011222079,
 502.770723085478,
 238.75901910327374,
 1391.0202139329165,
 538.1454654283822,
 322.2380396603607,
 10518.369083070755,
 1059.3089833147824,
 2592.2845806255937,
 5185.784757831693,
 653.2522977134213]
  ion_types = [0 for x in intensities]
  positions = [0 for x in intensities]
  return mzs, intensities, ion_types, positions

if __name__ == '__main__':
    app.run(host='0.0.0.0')
