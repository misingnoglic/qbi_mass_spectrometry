# Code to serve website

from flask import Flask, render_template, request, jsonify
from parse_data import amino_acid_name_parse
app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/graph')
def graph():
    return render_template("html_graph_test.html")



@app.route('/output_json')
def output_json():
    protein_name = request.args.get('protein_name')
    has_beginning_charge, name_one_hot_encoded, charge = amino_acid_name_parse(protein_name)
    mz_data, intensities = get_prediction(has_beginning_charge, name_one_hot_encoded, charge)
    d = {
        'protein_name': protein_name,
        'mz_data': mz_data,
        'intensities': intensities
    }
    return jsonify(d)


def get_prediction(has_beginning_charge, name_one_hot_encoded, charge):
    # Hardcoded for now until we can do actual ML
    mz_data = [129.102, 130.087, 143.082, 147.113, 155.081, 185.092, 214.119, 246.181, 256.129, 284.125, 327.166,
               377.222,398.204, 444.728, 471.244, 512.233, 605.329, 643.271, 760.883, 784.392, 791.395, 803.738,
               810.906, 840.419, 846.427, 860.433, 868.425, 870.104, 870.768, 881.943, 888.451, 893.782, 895.947,
               896.941, 917.463, 941.129, 966.969, 973.977, 1001.984, 1016.48, 1023.512, 1152.044]
    intensities = [794.5, 627.6, 612.5, 1266.7, 625.9, 4848.7, 455.6, 402.9, 6812.2, 672.6, 3782.6, 543.9, 788.0,
                   10000.0, 497.4, 423.0, 499.9, 572.2, 1181.6, 613.7, 1795.6, 5255.3, 822.1, 872.3, 5962.4, 1464.9,
                   394.7, 7227.1, 1058.6, 888.3, 4881.6, 6741.1, 1359.3, 656.7, 1786.9, 607.3, 368.7, 3633.5, 506.2,
                   548.2, 5230.3, 438.2]
    return mz_data, intensities


if __name__ == '__main__':
    app.run(host='0.0.0.0')
