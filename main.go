package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"sort"
	"time"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	badger "github.com/dgraph-io/badger/v4"
)

type badgerLogger struct {
	log zerolog.Logger
}

func (l *badgerLogger) Errorf(f string, v ...interface{}) {
	l.log.Error().Msgf(f, v...)
}

func (l *badgerLogger) Warningf(f string, v ...interface{}) {
	l.log.Warn().Msgf(f, v...)
}

func (l *badgerLogger) Infof(f string, v ...interface{}) {
	l.log.Info().Msgf(f, v...)
}

func (l *badgerLogger) Debugf(f string, v ...interface{}) {
	l.log.Debug().Msgf(f, v...)
}

var textChunks = []string{
	"Hello, world!",
	"The quick brown fox jumps over the lazy dog.",
	"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
	"Nulla facilisi. Sed ut imperdiet nunc.",
	"Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Donec eget nunc.",
	"Vivamus auctor, nunc nec lacinia tincidunt, nunc nunc fermentum nunc, nec fermentum nunc nunc nec nunc.",
	"Error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.",
	"Error (2) Co-pilot, engage the hyperdrive!",
	"Error (3) Co-pilot, engage the hyperdrive!",
	"Error Co-pilot this is not sensible log messages!",
}

func getEmbedding(m textencoding.Interface, text string) ([]float64, error) {
	result, err := m.Encode(context.Background(), text, int(bert.MeanPooling))
	if err != nil {
		return nil, err
	}

	return result.Vector.Data().F64(), nil
}

func makeEmbeddings(db *badger.DB, m textencoding.Interface) error {
	if err := db.Update(func(txn *badger.Txn) error {

		fn := func(text string) error {
			embedding, err := getEmbedding(m, text)
			if err != nil {
				return err
			}

			buf := bytes.NewBuffer(make([]byte, 0, len(embedding)*8))
			if err := binary.Write(buf, binary.LittleEndian, embedding); err != nil {
				return err
			}

			return txn.Set(buf.Bytes(), []byte(text))
		}

		for _, chunk := range textChunks {
			if err := fn(chunk); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return err
	}

	return db.View(func(txn *badger.Txn) error {
		it := txn.NewIterator(badger.DefaultIteratorOptions)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()

			var valCopy []byte
			if err := item.Value(func(val []byte) error {
				valCopy = append([]byte{}, val...)
				return nil
			}); err != nil {
				return err
			}

			buf := bytes.NewBuffer(item.KeyCopy(nil))
			key := make([]float64, buf.Len()/8)
			if err := binary.Read(buf, binary.LittleEndian, key); err != nil {
				return err
			}

			log.Info().Msgf("Inserted key[:3]=%v, value=%s", key[:3], valCopy)
		}

		return nil
	})

}

type Ranked struct {
	Rank float64
	Key  []float64
}

func cosineSimilarity(a, b []float64) float64 {
	dotProduct := 0.0
	magnitudeA := 0.0
	magnitudeB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}

	magnitudeA = 1.0 / (magnitudeA * magnitudeB)
	magnitudeB = 1.0 / (magnitudeA * magnitudeB)

	return dotProduct * magnitudeA * magnitudeB
}

func rankNearest(db *badger.DB, m textencoding.Interface, query string) error {
	target, err := getEmbedding(m, query)
	if err != nil {
		return err
	}

	ranked := make([]Ranked, 0, len(textChunks))

	if err := db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false
		it := txn.NewIterator(opts)
		defer it.Close()

		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()

			buf := bytes.NewBuffer(item.KeyCopy(nil))
			key := make([]float64, buf.Len()/8)
			if err := binary.Read(buf, binary.LittleEndian, key); err != nil {
				return err
			}

			ranked = append(ranked, Ranked{
				Rank: cosineSimilarity(target, key),
				Key:  key,
			})
		}

		return nil
	}); err != nil {
		return err
	}

	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].Rank > ranked[j].Rank
	})

	var nearest []byte
	if err := db.View(func(txn *badger.Txn) error {
		buf := bytes.NewBuffer(make([]byte, 0, len(target)*8))
		err := binary.Write(buf, binary.LittleEndian, ranked[0].Key)
		if err != nil {
			return err
		}

		item, err := txn.Get(buf.Bytes())
		if err != nil {
			return err
		}

		if err := item.Value(func(val []byte) error {
			nearest = append([]byte{}, val...)
			return nil
		}); err != nil {
			return err
		}

		return nil
	}); err != nil {
		return err
	}

	log.Info().Msgf("Nearest to %s: %s", query, nearest)
	for i, r := range ranked {
		log.Info().Msgf("Rank %d: %f %v", i, r.Rank, r.Key[:4])
	}

	return nil
}

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	db, err := badger.Open(badger.DefaultOptions("./badger.db").WithLogger(&badgerLogger{log: log.Logger.With().Str("pkg", "badger").Logger()}))
	if err != nil {
		log.Error().Err(err).Msgf("Error opening Badger database")
	}
	defer db.Close()

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	go func() {
		for range ticker.C {
		again:
			err := db.RunValueLogGC(0.7)
			if err == nil {
				goto again
			}
		}
	}()

	m, err := tasks.Load[textencoding.Interface](&tasks.Config{
		ModelsDir: "./models",
		ModelName: textencoding.DefaultModel,
	})
	if err != nil {
		log.Fatal().Err(err).Msgf("Error loading model")
	}

	if err := makeEmbeddings(db, m); err != nil {
		log.Fatal().Err(err).Msgf("Error making embeddings")
	}

	if err := rankNearest(db, m, "A commonly used latin phrase as placeholder text"); err != nil {
		log.Fatal().Err(err).Msgf("Error ranking nearest")
	}
}
