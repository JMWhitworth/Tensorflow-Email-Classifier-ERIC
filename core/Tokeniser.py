import string

class Tokeniser:
    def __init__(self, characters:str=string.printable, paddingSize:int=20, paddingChar=0) -> None:
        self.characters = characters
        self.paddingSize = paddingSize
        self.paddingChar = paddingChar
        
        self.encoder, self.decoder = self.createEncoder(characters)

    def createEncoder(self, characters:str) -> tuple:
        encoder, decoder = {}, {}
        
        for i, char in enumerate(characters):
            i+=1 #Skips 0 for padding character
            encoder[char] = i
            decoder[i] = [char]
        
        return (encoder, decoder)

    def encode(self, input:str) -> list:
        input = str(input)
        encoded = []
        
        #Encode the characters
        for character in input:
            if character in self.characters:
                encoded.append(self.encoder[character])
            else:
                print(f"New character found: {character}")
                self.characters += character
                self.encoder, self.decoder = self.createEncoder(self.characters)
                encoded.append(self.encoder[character])
        
        #Pad & trim to correct length
        while len(encoded) < self.paddingSize:
            encoded.append(self.paddingChar)
        while len(encoded) > self.paddingSize:
            del encoded[-1]
        
        return encoded
    
    def decode(self, input:list) -> str:
        decoded = ""
        for character in input:
            if character != self.paddingChar:
                decoded += self.decoder[character][0]
        return decoded

    def normalise(self, input:list) -> list:
        return [item/len(self.characters) for item in input]
