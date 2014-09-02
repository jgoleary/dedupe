#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import collections
import itertools
import logging
import time
import dedupe.tfidf as tfidf
import core

logger = logging.getLogger(__name__)

    

class Blocker:
    '''Takes in a record and returns all blocks that record belongs to'''
    def __init__(self, 
                 predicates, 
                 stop_words = None,
                 num_cores = 8) :

        if stop_words is None :
            stop_words = defaultdict(set)

        self.predicates = predicates
        self.num_cores = num_cores
        self.stop_words = stop_words

        self.tfidf_fields = defaultdict(set)

        for full_predicate in predicates :
            for predicate in full_predicate :
                if hasattr(predicate, 'canopy') :
                    self.tfidf_fields[predicate.field].add(predicate)

    #@profile
    def __call__(self, records):
        logger.info("calling blocker!!!")
        start_time = time.time()
        predicates = [(':' + str(i), predicate)
                      for i, predicate
                      in enumerate(self.predicates)]

        if self.num_cores < 2 :
            from multiprocessing.dummy import Process, Pool, Queue
        else :
            from backport import Process, Pool, Queue

        record_queue = Queue()
        result_queue = Queue()

        record_processor = RecordProcessor(predicates)

        n_map_processes = max(self.num_cores-1, 1)
        map_processes = [Process(target=record_processor,
                                 args=(record_queue,
                                       result_queue))
                         for _ in xrange(n_map_processes)]
        [process.start() for process in map_processes]

        core.fillQueue(record_queue, records, n_map_processes, chunk_size = 100)
        
        for bkri in _generateBlockRecord(result_queue, n_map_processes):
            yield bkri

        [process.join() for process in map_processes]
        logger.info("finished blocker call")



    def _resetCanopies(self) :
        # clear canopies to reduce memory usage
        for predicate_set in self.tfidf_fields.values() :
            for predicate in predicate_set :
                predicate.canopy = {}
                #if predicate._index is not None :
                #    predicate.index = None
                #    predicate.index_to_id = None


def _generateBlockRecord(result_queue, n_map_processes):
    seen_signals, i = 0, 0

    while seen_signals < n_map_processes:
        logger.info("waiting on next chunk return")
        block_keys_record_ids = result_queue.get()
        if block_keys_record_ids is None:
            seen_signals += 1
            continue
            
        for bkri in block_keys_record_ids :
            yield bkri
            
        i += 1
        if i and i % 10000 == 0 :
            logger.info('%(iteration)d, %(elapsed)f2 seconds',
                        {'iteration' :i,
                         'elapsed' :time.time() - start_time})


import os
class RecordProcessor(object) :
    
    def __init__(self, predicates):
        logging.info("initing with predicates %s" % predicates)
        self.predicates = predicates

    def __call__(self, record_queue, result_queue):
        logging.info("%d calling record processor" % os.getpid())

        while True :
            logging.info("%d waiting for next chunk get" % os.getpid())
            records = record_queue.get()
            #sentinel received, we won't get any more records
            if records is None:
                logging.info("%d got poison pill" % os.getpid())
                break

            logging.info("%d got %d records" % (os.getpid(), len(records)))
            self.applyPredicates(records, result_queue)
            
        #add sentinel to indicate this processor won't be sending any
        #more results
        result_queue.put(None)
        logging.info("%d exiting process call" % os.getpid())

    def applyPredicates(self, records, result_queue):
        block_keys_record_ids = []

        for (record_id, instance) in records :
            for pred_id, predicate in self.predicates :
                block_keys = predicate(record_id, instance)
                for block_key in block_keys :
                    block_keys_record_ids.append((block_key + pred_id, record_id))

        result_queue.put(block_keys_record_ids)
        logging.info("%d done putting" % os.getpid())

class DedupeBlocker(Blocker) :

    def tfIdfBlock(self, data, field): 
        '''Creates TF/IDF canopy of a given set of data'''

        indices = {}
        for predicate in self.tfidf_fields[field] :
            index = tfidf.TfIdfIndex(field, self.stop_words[field])
            indices[predicate] = index

        base_tokens = {}

        for record_id, doc in data :
            base_tokens[record_id] = doc
            for index in indices.values() :
                index.index(record_id, doc)

        logger.info(time.asctime())                

        for predicate in self.tfidf_fields[field] :
            logger.info("Canopy: %s", str(predicate))
            index = indices[predicate]
            predicate.canopy = index.canopy(base_tokens, 
                                            predicate.threshold)
        
        logger.info(time.asctime())                
               
class RecordLinkBlocker(Blocker) :
    def tfIdfIndex(self, data_2, field): 
        '''Creates TF/IDF index of a given set of data'''
        predicate = next(iter(self.tfidf_fields[field]))

        index = predicate.index
        canopy = predicate.canopy

        if index is None :
            index = tfidf.TfIdfIndex(field, self.stop_words[field])
            canopy = {}

        for record_id, doc in data_2  :
            index.index(record_id, doc)
            canopy[record_id] = (record_id,)

        for predicate in self.tfidf_fields[field] :
            predicate.index = index
            predicate.canopy = canopy

    def tfIdfUnindex(self, data_2, field) :
        '''Remove index of a given set of data'''
        predicate = next(iter(self.tfidf_fields[field]))

        index = predicate.index
        canopy = predicate.canopy

        for record_id, _ in data_2 :
            if record_id in canopy :
                index.unindex(record_id)
                del canopy[record_id]

        for predicate in self.tfidf_fields[field] :
            predicate.index = index
            predicate.canopy = canopy

