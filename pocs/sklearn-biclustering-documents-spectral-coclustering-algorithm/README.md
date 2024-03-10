### About

* `Spectral Co-clustering` algorithm
* Applied to twenty newsgroups dataset
* The best biclusters are determined by their normalized cut
* Best words are determined by comparing their sums inside and outside the bicluster
* For comparison docs are also clustered with `MiniBatchKMeans`

### Result
```
Vectorizing...
Coclustering...
Done in 4.76s. V-measure: 0.4415
MiniBatchKMeans...
Done in 2.23s. V-measure: 0.3015

Best biclusters:
----------------
bicluster 0 : 8 documents, 6 words
categories   : 100% talk.politics.mideast
words        : cosmo, angmar, alfalfa, alphalpha, proline, benson

bicluster 1 : 1948 documents, 4325 words
categories   : 23% talk.politics.guns, 18% talk.politics.misc, 17% sci.med
words        : gun, guns, geb, banks, gordon, clinton, pitt, cdt, surrender, veal

bicluster 2 : 1259 documents, 3534 words
categories   : 27% soc.religion.christian, 25% talk.politics.mideast, 25% alt.atheism
words        : god, jesus, christians, kent, sin, objective, belief, christ, faith, moral

bicluster 3 : 775 documents, 1623 words
categories   : 30% comp.windows.x, 25% comp.sys.ibm.pc.hardware, 20% comp.graphics
words        : scsi, nada, ide, vga, esdi, isa, kth, s3, vlb, bmug

bicluster 4 : 2180 documents, 2802 words
categories   : 18% comp.sys.mac.hardware, 16% sci.electronics, 16% comp.sys.ibm.pc.hardware
words        : voltage, shipping, circuit, receiver, processing, scope, mpce, analog, kolstad, umass
```