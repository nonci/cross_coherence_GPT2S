import csv, sys

with open(sys.argv[1], 'r') as ifile:
	reader = csv.reader(ifile, delimiter=',', quotechar='"')
	group = []
	prev = ''
	for row in reader:
		if row[1]==prev:
			group[-1].append(row)
		else:
			group.append([row])
		prev = row[1]
		
sorted_groups = sorted(group[1:], key = lambda _: len(_), reverse=True)

with open(sys.argv[2], 'w') as ofile:
	writer = csv.writer(ofile, delimiter=',', quoting=csv.QUOTE_MINIMAL, quotechar='"')
	writer.writerow(group[0][0])
	for g in sorted_groups:
		for l in g:
			writer.writerow(l)
