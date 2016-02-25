import json

infile = './content_type_id_mapping.json'
outfile = './ontology_map.json'

with open(infile) as i:
    map_list = json.load(i)

map_dict = {}
for count, ct in enumerate(map_list):
    map_dict[str(count)] = ct

print "map_list={}, map_dict={}".format(map_list, map_dict)

with open(outfile, 'w') as i:
    json.dump(map_dict, i)
