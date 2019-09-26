import nltk
from wordcloud import WordCloud

text = '''
Super Smash Bros.[a] is a series of crossover fighting video games published by Nintendo, and primarily features characters from various Nintendo franchises. The series was created by Masahiro Sakurai, who has directed every game in the series. The gameplay objective differs from that of traditional fighters in that the aim is to knock opponents off the stage instead of depleting life bars.

The original Super Smash Bros. was released in 1999 for the Nintendo 64. The series achieved even greater success with the release of Super Smash Bros. Melee, which was released in 2001 for the GameCube and became the bestselling game on that system. A third installment, Super Smash Bros. Brawl, was released in 2008 for the Wii. Although HAL Laboratory had been the developer of the first two games, the third game was developed through the collaboration of several companies. The fourth installment, Super Smash Bros. for Nintendo 3DS and Wii U, were released in 2014 for the Nintendo 3DS and Wii U, respectively. The 3DS installment was the first for a handheld platform. A fifth installment, Super Smash Bros. Ultimate, was released in 2018 for the Nintendo Switch.

The series features many characters from Nintendo's most popular franchises, including Super Mario, Donkey Kong, The Legend of Zelda, Metroid, Star Fox, Kirby, Yoshi and Pokémon. The original Super Smash Bros. had only 12 playable characters, with the roster count rising for each successive game and later including third-party characters, with Ultimate containing every character playable in the previous games. Some characters are able to transform into different forms that have different styles of play and sets of moves. Every game in the series has been well received by critics, with much praise given to their multiplayer features, spawning a large competitive community that has been featured in several gaming tournaments.
'''

#ストップワードを設定
stop_words = nltk.corpus.stopwords.words('english')

# words = nltk.word_tokenize(text)

wordcloud = WordCloud(min_font_size=5, collocations=False, background_color='black',
                      width=900, height=500,stopwords=set(stop_words),max_words=100)
wordcloud.generate(text).to_file("../result/wordcloud/sample.png")
