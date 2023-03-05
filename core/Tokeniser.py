class Tokeniser:
    def __init__(self, paddingSize:int=20, paddingChar=0) -> None:
        self.characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&'()*+,-./:;<=>?@[\]^_`{|}~"+'"'
        self.paddingSize = paddingSize
        self.paddingChar = paddingChar
        
        self.encoder, self.decoder = self.createEncoder(self.characters)
    
    def createEncoder(self, characters:str) -> tuple:
        encoder, decoder = {}, {}
        
        for i, char in enumerate(characters):
            i+=1 #Skips 0, as its reserved for padding character
            encoder[char] = i
            decoder[i] = [char]
        
        return (encoder, decoder)
    
    def encode(self, item:str) -> list:
        item = self.normalise(item)
        encoded = []
        
        for character in item:
            if character in self.characters:
                encoded.append(self.encoder[character])
            else:
                print(f"New character found: {character}")
                self.characters += character
                self.encoder, self.decoder = self.createEncoder(self.characters)
                encoded.append(self.encoder[character])
        
        encoded = self.padd(encoded)
        return encoded
    
    def decode(self, item:list) -> str:
        decoded = ""
        for character in item:
            if character != self.paddingChar:
                decoded += self.decoder[character][0]
        return self.normalise(decoded)
    
    def normalise(self, item:str) -> str:
        item = str(item)
        
        #Fixes issue where multiple types of new lines are used
        item = "".join(l for l in item.splitlines() if l)
        
        #Remove undesired characters
        item = item.lower().replace('\n', '').replace(' ','').replace(' ','')
        
        return item.strip()
    
    def padd(self, item:list) -> list:
        while len(item) < self.paddingSize:
            item.append(self.paddingChar)
        while len(item) > self.paddingSize:
            del item[-1]
        return item
