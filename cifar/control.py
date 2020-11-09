import os
import datetime

import flask

import onnxds.read_dataset as helper

from aliaser import AliasService
from cifar import PersistentCifarModel

default_oxfilename = 'controlserver.onnx'

class ControlServer:
    def __init__(self, logger, oxfile,
        save_dir = '/tmp/'):
        self.app = flask.Flask(__name__)

        first_entry = next(helper.load(oxfile))
        raw_inshape = list(first_entry['image'].shape[1:])

        self.alias_svc = AliasService(logger=logger)
        self.cnn = PersistentCifarModel(self.alias_svc, raw_inshape, 15121)

        @self.app.route('/api/v1', methods=["GET"])
        def info_view():
            """List of routes for this API."""
            return flask.jsonify({
                'info': 'GET /api/v1',
                'serialize': 'POST /api/v1/serialize',
                'shutdown': 'POST /api/v1/shutdown',
            })

        @self.app.route('/api/v1/serialize', methods=["POST"])
        def serialize():
            """Serialize NN controlled by this app."""
            config = flask.request.get_json()
            if config is None:
                config = {}
            save_file = config.get('savefile', None)
            if save_file is None:
                save_file = default_oxfilename
            target_dir = os.path.join(save_dir, str(datetime.datetime.now().isoformat()))
            os.mkdir(target_dir)
            target_filepath = os.path.join(target_dir, save_file)
            if cnn.save(target_filepath):
                msgfmt = 'Saved model to "{}"'
            else:
                msgfmt = 'Failed to save model to "{}"'
            logger.info(msgfmt)
            return flask.jsonify({
                "msg": msgfmt.format(target_filepath)
            })

        @self.app.route('/api/v1/shutdown', methods=["POST"])
        def shutdown():
            """Shutdown NN controlled by this app."""
            stopfunc = flask.request.environ.get('werkzeug.server.shutdown')
            if stopfunc is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            msg = "Server shutting down..."
            logger.info(msg)
            stopfunc()
            return flask.jsonify({
                "msg": msg
            })

    def serve(self, port):
        self.app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    train_oxfile = 'models/cifar_train.onnx'
    port = 12348

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    server = ControlServer(logger, train_oxfile)
    server.serve(port)
