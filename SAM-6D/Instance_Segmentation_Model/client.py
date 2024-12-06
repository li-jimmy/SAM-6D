import time
import httpx

# URL of the FastAPI server
SERVER_URL = "http://localhost:48001/segment_and_register/"

# Paths to the images
rgb_image_path = "/home/ubuntu/tao-experiments/data/rgbd_in_three_mustard/0_color.png"
depth_image_path = "/home/ubuntu/tao-experiments/data/rgbd_in_three_mustard/0_depth.png"

def make_request(client, rgb_path, depth_path):
    with open(rgb_path, "rb") as rgb_file, open(depth_path, "rb") as depth_file:
        # Create the files payload
        files = {
            "rgb": ("rgb_image.jpg", rgb_file, "image/jpeg"),
            "depth": ("depth_image.png", depth_file, "image/png"),
        }
        start_time = time.time()
        # Send the POST request
        response = client.post(SERVER_URL, files=files)

        # Check and print the response
        if response.status_code == 200:
            print("Response received:", response.json())
        else:
            print(f"Error: {response.status_code} - {response.text}")
        print('Request took:', time.time() - start_time)

if __name__ == "__main__":
    http_client = httpx.Client(http2=True, verify=False)
    import ipdb; ipdb.set_trace()
    make_request(http_client, rgb_image_path, depth_image_path)
    http_client.close()