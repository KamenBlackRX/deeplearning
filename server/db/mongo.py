# encoding utf-8
import cv2
import numpy as np
from gridfs import GridFS
from matplotlib import pyplot as plt
from pymongo import MongoClient


class Mongo:
    """
    Class that wraps PyMongo for easier access to operations.
    """

    def __init__(self, **kwargs):
        """
        Initializes Mongo database

        Keyword Arguments:
            :param host: host that you want to access, default = 'localhost'
            :type host: str
            :param port -- port that you want to access, default = 27017
            :type port: int
            :param namespace: gridfs namespace.
            :type namespace: str
        """
        host = 'localhost'
        port = 27017
        namespace = 'IMG_Holder'
        if kwargs is not None:
            if 'host' in kwargs.keys():
                host = kwargs['host']
            if 'port' in kwargs.keys():
                port = kwargs['port']
            if 'namespace' in kwargs.keys():
                namespace = kwargs['namespace']
        self.client = MongoClient(host, port)
        self.db = self.client['default']
        self.grid = GridFS(self.db, namespace)

    def __del__(self):
        """
        Disconnect a active connection , Called in GC
        """
        self.client.close()

    def set_working_database(self, database):
        """
        Sets the working database

        Arguments:
            database {String} -- name of the working database, creates one if none exists.
        """
        self.db = self.client[database]

    def create(self, collection, data):
        """
        Insert data to the database.

        Arguments:
            :param collection: name of the collection you want to insert to.
            :type collection: str
            :param data: dictionary containing the information you want to insert at the database.
            :type data: dict or list
        """
        assert type(data) is list or type(data) is dict, "data must be either a list of dictionaries or a dictionary."
        if type(data) is list:
            multi = True
        elif type(data) is dict:
            multi = False

        if multi is False:
            self.db[collection].insert_one(data)
        else:
            self.db[collection].insert_many(data)

    def insert_grid(self, **kwargs):
        """
        Insert file to grid.

        Keyword Arguments:
             :param file: path to the file.
             :type file: str
             :param filename: name to the file for easier search
             :type filename: str
        """
        for entry in self.read('control'):
            id = entry['image_id'] + 1
            entry_id = entry['_id']
            break
        name = ''
        file = ''
        owner = ''
        if kwargs is not None:
            if 'file' in kwargs.keys():
                file = kwargs['file']
            if 'filename' in kwargs.keys():
                name = kwargs['filename']
            if 'id' in kwargs.keys():
                id = kwargs['id']
            if 'owner' in kwargs.keys():
                owner = kwargs['owner']
            try:
                with open(file, 'rb') as f:
                    self.grid.put(f, filename=name, _id=id, owner=owner)
                    self.update('control', {'_id': entry_id}, {'$set': {'image_id': id}})
            except IOError:
                print('File doesnt exist.')

    def read(self, collection, **kwargs):
        """
        Read from database, if query is defined, use it to filter through all elements.

        Arguments:
            collection {String} -- name of the collection you want to read.

        Keyword Arguments:
            query {Dictionary} -- dictionary containing the filter for the documents you want to read.
            verbose {Boolean} -- flag that returns a list of read documents instead of printing it.

        Returns:
            List -- may return list of read documents if verbose is set to True.
        """
        query = {}
        if 'query' in kwargs.keys():
            query = kwargs['query']
        read = self.db[collection].find(query)
        if 'verbose' not in kwargs.keys() or kwargs['verbose'] is not True:
            return read
        elif 'verbose' in kwargs.keys() and kwargs['verbose'] is True:
            print([docs for docs in read])

    def read_grid(self, query, **kwargs):
        """
        Read from grid and return all images found.

        Arguments:
            :param query: filter to search for, usually with filename as key.
            :type query: dict

        Keyword Arguments:
            :param show: shows the image.
            :type show: bool

        Returns:
            :return opencv image
            :rtype: np.ndarray
        """
        show = False
        if 'show' in kwargs.keys():
            show = kwargs['show']
        data = []
        cv_image = None
        for f in self.grid.find(query):
            data.append(f.read())

        for image in data:
            np_image = np.fromstring(image, np.uint8)
            cv_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            if show is True:
                plt.imshow(cv_image)
                plt.show()

        return cv_image

    def update(self, collection, query, change, **kwargs):
        """
        Updates the documents that match query to change dictionary.

        Arguments:
            collection {String} -- name of the collection you want to update.
            query {Dictionary} -- dictionary containing the filter for the documents you want to update.
            change {Dictionary} -- dictionary containing the operation you want to perform and the properties of the
            documents that you want to update.

        Keyword Arguments:
            log {Boolean} -- return the log of the function.
            multi {Boolean} -- update multiple documents.
            upsert {Boolean} -- if no document match the filter, insert the update to the database.

        Returns:
            Dictionary -- returns the log of the update if log is set to True, it contains the number of documents that
            matched the query and the number of documents updated.
        """
        multi = False
        upsert = False
        log = False
        if kwargs is not None:
            if 'multi' in kwargs.keys():
                multi = kwargs['multi']
            if 'upsert' in kwargs.keys():
                upsert = kwargs['upsert']
            if 'log' in kwargs.keys():
                log = kwargs['log']

        if multi is not True:
            result = self.db[collection].update_one(
                query, change, upsert=upsert)
        else:
            result = self.db[collection].update_many(
                query, change, upsert=upsert)
        if log is True:
            return result

    def delete(self, collection, query, **kwargs):
        """
        Deletes the documents that match query filter.

        Arguments:
            collection {String} -- name of the collection you want to delete.
            query {Dictionary} -- dictionary containing the filter for the documents you want to delete.

        Keyword Arguments:
            log {Boolean} -- return the log of the function.
            multi {Boolean} -- delete multiple documents. 
        """
        multi = False
        log = False
        if kwargs is not None:
            if 'multi' in kwargs.keys():
                multi = kwargs['multi']
            if 'log' in kwargs.keys():
                log = kwargs['log']

        if multi is False:
            result = self.db[collection].delete_one(query)
        else:
            result = self.db[collection].delete_many(query)

        if log is True:
            return result

    def delete_from_grid(self, f_id):
        """
        Deletes a file from grid.
        Arguments:
            :param id: id of file.
            :type id: int
        """
        self.grid.delete(f_id)
