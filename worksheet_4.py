#q1
# import CSVProcessor
# from CSVProcessor import *
# load()
# rows()
# cols()
# missing()

#q4
# import re
# pattern = 'GAATTC'
# pattern1 = 'ATCC'
# pattern2 = 'GC'
# dna = 'ATCGCGAATTCAC'
# match = re.search(pattern2, dna)
# print(match)


#q5
# import re
# pattern1 = 'GGACC' and 'GGTCC'
# pattern2 = 'GGACC' and 'AATTC'
# dna = 'ATCGCGAATTCAC'
# match = re.search(pattern2, dna)
# print(match)


#6
# import re
# dna = "ATCGCGAATTCAC"
# pattern1 = 'GC+[ACGT]+GC'
# pattern2 = 'GC+[ACGT]+AAT'
# match = re.search(pattern2, dna)
# print(match)


#7
#     ^ symbol denotes the start of a sequence/pattern.
#     Characters inside [] brackets denotes any one character to be matched whichever is present.
#     Numbers inside {} denotes matching the preceding character or pattern from n to m times.
#     $ symbol denote end of any pattern.
#
# Therefore, the pattern will match mRNA sequences that start with AUG (start codon)
# followed by a coding region of 30 to 1000 bases consisting of A, U, G, and C, and
# ending with a poly-A tail of 5 to 10 consecutive 'A' characters.


#8
# dna = "ATCGCGYAATTCAC"
# for i in dna:
#     if i not in {'A', 'T', 'G', 'C'}:
#         print(i)


#9
# import re
# scientific_name = "Homo sapiens"
# #scientific_name = "Drosophila melanogaster"
# tokens = scientific_name.split(' ')
# gen_sp1 = re.compile(tokens[0])
# print("Genus is",  gen_sp1)
# gen_sp2 = re.compile(tokens[1])
# print("and species is", gen_sp2)


#10
# import re
# dna = "CGATNCGGAACGATC"
# match = re.findall('[^ATGC]', dna)
# print(match)
#
# pattern = re.compile('[^ATGC]')
# m = pattern.finditer(dna)
# pos = [matches.start() for matches in m]
# print(pos)


#11
# import re
# dna = 'ACTGCATTATATCGTACGAAATTATACGCGCG'
# p = ['ATTATAT','ATTATAT']
# pattern = '|'.join(p)
# match = re.findall(pattern, dna)
# print(match)


#12
# import re
# dna = "ACTNGCATRGCTACGTYACGATSCGAWTCG"
# pattern = re.split('[^ATCG]', dna)
# print(pattern)


#13
import re

accessions = ['xkn59438', 'yhdck2', 'eihd39d9', 'chdsye847', 'hedle3455', 'xjhd53e', '45da', 'de37dp']

# p = re.compile(r'.*5.*')
# for i in accessions:
#     if p.match(i):
#         print(i)


# p = re.compile(r'.*[de].*')
# for i in accessions:
#     if p.match(i):
#         print(i)


# p = re.compile(r'.*de.*')
# for i in accessions:
#     if p.match(i):
#         print(i)

# p = re.compile(r'.*d[a-z]e.*')
# for i in accessions:
#     if p.match(i):
#         print(i)

# p = re.compile(r'(.*d.*)(.*e.*)')
# for i in accessions:
#     if p.match(i):
#         print(i)

# p = re.compile(r'^[xy]')
# for i in accessions:
#     if p.match(i):
#         print(i)


# p = re.compile(r'^[xy].*e$')
# for i in accessions:
#      if p.match(i):
#          print(i)


# p = re.compile(r'.*\d{3}.*')
# for i in accessions:
#     if p.match(i):
#         print(i)


# p = re.compile(r'.*d[arp]$')
# for i in accessions:
#      if p.match(i):
#          print(i)


