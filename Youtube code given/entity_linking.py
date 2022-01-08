import os
import sys
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

key = "06b2016b1dcc469ba4ced35762c30d48"
endpoint = "https://abcddsadadsda.cognitiveservices.azure.com/"


def authenticate_client():
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, credential=ta_credential)
    return text_analytics_client


def entity_linking_example(client, vid_dir, out_dir):

	for fname in os.listdir(vid_dir):
		if not os.path.exists(os.path.join(out_dir, fname)):
			os.mkdir(os.path.join(out_dir, fname))
			for shot in os.listdir(os.path.join(vid_dir, fname)):
				with open(os.path.join(vid_dir, fname, shot)) as f:
					documents = f.read().strip()
				f.close()
				if len(documents) > 0:
					result = client.recognize_linked_entities(documents = [documents])[0]

					print("Linked Entities in " + fname + "_" + shot)
					outf = open(os.path.join(out_dir, fname, shot) , 'wb')
					for entity in result.entities:
						outf.write(("Name: " + entity.name + "\tId: " + entity.data_source_entity_id+ "\tUrl: "+ entity.url+
						"\n\tData Source: "+ entity.data_source + "\n").encode())
						outf.write(("Matches:\n").encode())
						for match in entity.matches:
							outf.write(("\tText:"+ match.text).encode())
							outf.write(("\tConfidence Score: {0:.2f} \n".format(match.confidence_score)).encode())
				else:
					print("empty transcript file!")                
			# break    

client = authenticate_client()
vid_dir = sys.argv[1]
out_dir = sys.argv[2]
entity_linking_example(client, vid_dir, out_dir)
