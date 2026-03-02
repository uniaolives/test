package chain1337

type Client struct {
	Url string
}

func NewClient(url string) *Client {
	return &Client{Url: url}
}

func (c *Client) VerifyAttestation(source, dest, proof string) (bool, error) {
	return true, nil
}
