#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"

#include "caffe/util/db.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using boost::scoped_ptr;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert an input format to another\n"
        "Supported formats are leveldb, lmdb and datum file.\n"
        "Usage:\n"
        "    convert_format INPUT_TYPE INPUT_PATH OUTPUT_TYPE OUTPUT_PATH\n");

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_format");
    return 1;
  }

  // Create input DB
  scoped_ptr<db::DB> db_input(db::GetDB(argv[1]));
  db_input->Open(argv[2], db::READ);
  scoped_ptr<db::Cursor> cur(db_input->NewCursor());

  // Create output DB
  scoped_ptr<db::DB> db_output(db::GetDB(argv[3]));
  db_output->Open(argv[4], db::NEW);
  scoped_ptr<db::Transaction> txn(db_output->NewTransaction());

  while (cur->valid()) {
    txn->Put(cur->key(), cur->value());
    cur->Next();
  }
  txn->Commit();
  return 0;
}
