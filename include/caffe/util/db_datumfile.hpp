#ifndef CAFFE_UTIL_DB_DATUMFILE_HPP
#define CAFFE_UTIL_DB_DATUMFILE_HPP

#include <string>

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

#define MAX_BUF 104857600  // max entry size

class DatumFileCursor : public Cursor {
 public:
  explicit DatumFileCursor(const string& path) {
    this->path_ = path;
    in_ = NULL;
    SeekToFirst();
  }
  virtual ~DatumFileCursor() {
    if (in_ != NULL && in_->is_open()) {
      in_->close();
      delete in_;
      in_ = NULL;
    }
  }
  virtual void SeekToFirst();

  virtual void Next();

  virtual string key() {
    CHECK(valid()) << "not valid state at key()";
    return key_;
  }
  virtual string value() {
    CHECK(valid()) << "not valid state at value()";
    return value_;
  }

  virtual bool valid() { return valid_; }

 private:
  string path_;
  std::ifstream* in_;
  bool valid_;

  string key_, value_;
};

class DatumFileTransaction : public Transaction {
 public:
  explicit DatumFileTransaction(std::ofstream* out) {
    this->out_ = out;
  }

  virtual void Put(const string& key, const string& value);

  virtual void Commit() {
    out_->flush();
  }

 private:
  std::ofstream* out_;
  DISABLE_COPY_AND_ASSIGN(DatumFileTransaction);
};


class DatumFileDB : public DB {
 public:
  DatumFileDB() { out_ = NULL; can_write_ = false;}
  virtual ~DatumFileDB() { Close(); }
  virtual void Open(const string& source, Mode mode) {
    path_ = source;
    this->can_write_ = mode != db::READ;
  }
  virtual void Close() {
    if (out_ != NULL) {
      out_->close();
      delete out_;
      out_ = NULL;
    }
  }
  virtual DatumFileCursor* NewCursor() {
    return new DatumFileCursor(this->path_);
  }
  virtual Transaction* NewTransaction();

 private:
  string path_;
  std::ofstream* out_;

  bool can_write_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_DATUMFILE_HPP
