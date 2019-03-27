import datetime
import simplejson
import urllib.request

from user_definition import *

try:
    output_file = open('~/'+git_repo_name + "/" + output_file_name, "a")
except FileNotFoundError:
    output_file = open('~/'+git_repo_name+"/"+output_file_name, "w+")
# https://developers.google.com/maps/documentation/javascript/get-api-key
apikey = 'AIzaSyDr9YOisPC4G_HVRquGYihYvoi6uTgx65M'
url = "https://maps.googleapis.com/maps/api/distancematrix/json?" \
      "key={0}&origins={1}&destinations={2}&mode=driving&departure_time=now" \
      "&language=en-EN&sensor=false".\
    format(str(apikey), str(orig_coord), str(dest_coord))
result = simplejson.load(urllib.request.urlopen(url))
driving_time = result['rows'][0]['elements'][0]['duration_in_traffic']['text']

output_file.write(str(datetime.datetime.now()) + "\n")
output_file.write(result['origin_addresses'][0] + "\n")
output_file.write(result['destination_addresses'][0] + "\n")
output_file.write(driving_time + "\n\n")
output_file.close()
