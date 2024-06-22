# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------
Copyright 2023 Benjamin Alexander Albert [Alejandro Chavez Lab at UCSD]
All Rights Reserved
TODO License
tensorcache.py
--------------------------------------------------------------------------------

A class to manage a cache directory in which each file is a serialized Tensor.

Within the cache directory, there is a file called cachemap.csv; each line of
the cachemap contains two tokens, delimited by a comma:
    [0] : key (e.g. amino acid sequence)
    [1] : UUID that is also the name of the cache file for the given key.
          This is simply a random alphanumeric (hex) string so that
          we can save keys that may otherwise have undesirable file names.

"""

import os
import uuid
import torch


class TensorCache:
    def __init__(self, cachedir : str):
        """
        Construct a TensorCache with a cachedir that may or may not exist.

        Parameters
        ----------
        cachedir : str
            Absolute path to a either an existing cache dir
            the path at which a new cache dir will be created.
        """
        self._cachedir   = cachedir
        self._cachemapfp = os.path.join(cachedir, "cachemap.csv")
        self._cachemap   = dict()
        if os.path.exists(self._cachemapfp):
            self._cachemap = self._readCacheMap()

    def _makeUUID(self) -> str:
        """
        Create a new hex UUID using uuid4() that
        does not already exist in the cachedir.
        """
        uuidhex = None
        while uuidhex is None or \
              os.path.exists(os.path.join(self._cachedir, uuidhex)):
            uuidhex = uuid.uuid4().hex
        return uuidhex

    def _readCacheMap(self) -> dict:
        """
        Reads the serialized cachemap dict pointed to by self._cachemapfp.

        Inverse of _writeCacheMap()

        Returns
        -------
        dict mapping the first str token to second str token
        """
        cache = dict()
        with open(self._cachemapfp, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            delimidx = line.find(',')
            if delimidx > 0:
                cache[line[:delimidx]] = line[delimidx+1:]
        return cache
    
    def _writeCacheMap(self) -> None:
        """
        Writes the cachemap dict to the file pointed to by self._cachemapfp.

        Inverse of _readCacheMap()
        """
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)
        lines = [k+','+v for k,v in self._cachemap.items()]
        with open(self._cachemapfp, 'w') as f:
            f.write('\n'.join(lines))

    def _getTensorCachePath(self, key : str) -> str:
        """
        Get the file path to the cache mapped to by the key.

        Raises
        ------
        ValueError if the key is not cached.
        """
        if not self.isCached(key):
            raise ValueError("key {} is not cached".format(key))
        return os.path.join(self._cachedir, self._cachemap[key])



    def isCached(self, key : str) -> bool:
        if self._cachemap is None:
            return False
        return key in self._cachemap

    def read(self, key : str) -> torch.Tensor:
        return torch.load(self._getTensorCachePath(key)).cpu()

    def write(
        self,
        key : str,
        val : torch.Tensor) -> None:
        
        """
        Cache the given key-Tensor pair
        Parameters
        ----------

        key : str
            Key that maps to the corresponding Tensor.
            The mapping may be many-to-one.
            Each key is also assigned a UUID to create a desirable file name.

        val : Tensor
            The value mapped to by key, to be stored with torch.save(val)
        """
        self._cachemap[key] = self._makeUUID()
        self._writeCacheMap()
        torch.save(
            obj=val.cpu(),
            f=self._getTensorCachePath(key)
            )
