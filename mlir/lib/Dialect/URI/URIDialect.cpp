//===- URIDialect.cpp - URI MLIR Dialect Implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/URI/URIDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

#include "mlir/Dialect/URI/URIDialect.cpp.inc"

namespace {

class URIResourceManager {
public:
  /// The class represents an individual entry of a blob.
  class Entry {
  public:
    /// Return the key used to reference this blob.
    StringRef getKey() const { return key; }

    /// Return the external reference for this blob.
    StringRef getURI() const { return uri; }

    /// Return the blob owned by this entry if one has been initialized. Returns
    /// nullptr otherwise.
    const AsmResourceBlob *getBlob() const { return blob ? &*blob : nullptr; }
    AsmResourceBlob *getBlob() { return blob ? &*blob : nullptr; }

    /// Set the blob owned by this entry.
    void setBlob(AsmResourceBlob &&newBlob) { blob = std::move(newBlob); }

    /// Set the URI owned by this entry.
    void setURI(StringRef newURI) { uri = newURI; }

  private:
    Entry() = default;
    Entry(Entry &&) = default;
    Entry &operator=(const Entry &) = delete;
    Entry &operator=(Entry &&) = delete;

    /// Initialize this entry with the given key and blob.
    void initialize(StringRef newKey, StringRef newURI,
                    std::optional<AsmResourceBlob> newBlob) {
      key = newKey;
      uri = newURI;
      blob = std::move(newBlob);
    }

    /// The key used for this blob.
    StringRef key;

    /// The URI this resource refers to.
    std::string uri;

    /// The blob that is referenced by this entry if it is valid.
    std::optional<AsmResourceBlob> blob;

    /// Allow access to the constructors.
    friend URIResourceManager;
    friend class llvm::StringMapEntryStorage<Entry>;
  };

  /// Return the blob registered for the given name, or nullptr if no blob
  /// is registered.
  Entry *lookup(StringRef name);
  const Entry *lookup(StringRef name) const {
    return const_cast<URIResourceManager *>(this)->lookup(name);
  }

  void update(StringRef name, StringRef uri,
              std::optional<AsmResourceBlob> newBlob) {
    Entry *entry = lookup(name);
    assert(entry && "`update` expects an existing entry for the provided name");
    entry->setURI(uri);
    if (newBlob)
      entry->setBlob(std::move(*newBlob));
  }

  /// Insert a new entry with the provided name and optional blob data. The name
  /// may be modified during insertion if another entry already exists with that
  /// name. Returns the inserted entry.
  Entry &insert(StringRef name, StringRef uri,
                std::optional<AsmResourceBlob> blob = {});
  /// Insertion method that returns a dialect specific handle to the inserted
  /// entry.
  template <typename HandleT>
  HandleT insert(typename HandleT::Dialect *dialect, StringRef name,
                 StringRef uri, std::optional<AsmResourceBlob> blob = {}) {
    Entry &entry = insert(name, uri, std::move(blob));
    return HandleT(&entry, dialect);
  }

private:
  /// A mutex to protect access to the blob map.
  llvm::sys::SmartRWMutex<true> mapLock;

  /// The internal map of tracked blobs. StringMap stores entries in distinct
  /// allocations, so we can freely take references to the data without fear of
  /// invalidation during additional insertion/deletion.
  llvm::StringMap<Entry> map;
};

auto URIResourceManager::lookup(StringRef name) -> Entry * {
  llvm::sys::SmartScopedReader<true> reader(mapLock);

  auto it = map.find(name);
  return it != map.end() ? &it->second : nullptr;
}

URIResourceManager::Entry &
URIResourceManager::insert(StringRef name, StringRef uri,
                           std::optional<AsmResourceBlob> blob) {
  llvm::sys::SmartScopedWriter<true> writer(mapLock);

  // Functor used to attempt insertion with a given name.
  auto tryInsertion = [&](StringRef name) -> Entry * {
    auto it = map.try_emplace(name, Entry());
    if (it.second) {
      it.first->second.initialize(it.first->getKey(), uri, std::move(blob));
      return &it.first->second;
    }
    return nullptr;
  };

  // Try inserting with the name provided by the user.
  if (Entry *entry = tryInsertion(name))
    return *entry;

  // If an entry already exists for the user provided name, tweak the name and
  // re-attempt insertion until we find one that is unique.
  llvm::SmallString<32> nameStorage(name);
  nameStorage.push_back('_');
  size_t nameCounter = 1;
  do {
    Twine(nameCounter++).toVector(nameStorage);

    // Try inserting with the new name.
    if (Entry *entry = tryInsertion(nameStorage))
      return *entry;
    nameStorage.resize(name.size() + 1);
  } while (true);
}
} // namespace

/// A handle used to reference external elements instances.
struct mlir::uri::URIResourceHandle
    : public AsmDialectResourceHandleBase<
          URIResourceHandle, URIResourceManager::Entry, uri::URIDialect> {

  using AsmDialectResourceHandleBase<
      URIResourceHandle, URIResourceManager::Entry,
      uri::URIDialect>::AsmDialectResourceHandleBase;
  /// Return the human readable string key for this handle.
  StringRef getKey() const { return this->getResource()->getKey(); }

  /// Return the URI corresponding to this resource.
  StringRef getURI() const { return this->getResource()->getURI(); }

  /// Return the blob referenced by this handle if the underlying resource has
  /// been initialized. Returns nullptr otherwise.
  AsmResourceBlob *getBlob() { return this->getResource()->getBlob(); }
  const AsmResourceBlob *getBlob() const {
    return this->getResource()->getBlob();
  }
};

namespace {
class URIResourceManagerInterface
    : public DialectInterface::Base<ResourceBlobManagerDialectInterface> {
public:
  URIResourceManagerInterface(Dialect *dialect)
      : Base(dialect), manager(std::make_shared<URIResourceManager>()) {}

  uri::URIResourceHandle insert(StringRef name, StringRef uri = {},
                                std::optional<AsmResourceBlob> blob = {}) {
    return manager->insert<uri::URIResourceHandle>(
        static_cast<uri::URIDialect *>(getDialect()), name, uri,
        std::move(blob));
  }

  void update(StringRef name, StringRef uri,
              std::optional<AsmResourceBlob> newBlob) {
    manager->update(name, uri, std::move(newBlob));
  }

private:
  /// The blob manager owned by the dialect implementing this interface.
  std::shared_ptr<URIResourceManager> manager;
};

//===----------------------------------------------------------------------===//
// URIOpAsmInterface
//===----------------------------------------------------------------------===//

class URIOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;
  URIOpAsmInterface(Dialect *dialect, URIResourceManagerInterface &mgr)
      : OpAsmDialectInterface(dialect), manager(mgr) {}

  //===------------------------------------------------------------------===//
  // Resources
  //===------------------------------------------------------------------===//

  std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const override {
    return cast<uri::URIResourceHandle>(handle).getKey().str();
  }

  FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return manager.insert(key);
  }

  LogicalResult parseResource(AsmParsedResourceEntry &entry) const final {
    // If its a string, then treat it as a filename, else error.
    if (entry.getKind() != AsmResourceEntryKind::String)
      return failure();

    FailureOr<std::string> dialectAndId = entry.parseAsString();
    if (failed(dialectAndId))
      return failure();

    auto [dialectName, id] = StringRef(*dialectAndId).split(':');
    std::optional<AsmResourceBlob> blob;
    auto *dialect = getContext()->getOrLoadDialect(dialectName);
    if (dialect) {
      const auto *interface =
          dyn_cast<ExternalResourceDialectInterface>(dialect);
      if (interface) {
        if (failed(interface->load(id, *blob)))
          return entry.emitError() << "invalid external reference";
      }
    }

    manager.update(entry.getKey(), *dialectAndId, std::move(blob));
    return success();
  }

  void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &provider) const final {
    // Print referenced resources using external reference.
    for (const AsmDialectResourceHandle &handle : referencedResources) {
      if (const auto *dialectHandle =
              dyn_cast<uri::URIResourceHandle>(&handle)) {
        auto key = dialectHandle->getKey();
        if (dialectHandle->getURI().empty()) {
          // TODO: Emit error?
          continue;
        }
        provider.buildString(key, dialectHandle->getURI());
      }
    }
  }

private:
  /// The blob manager for the dialect.
  URIResourceManagerInterface &manager;
};
} // namespace

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/URI/URIDialectAttrDefs.cpp.inc"

void uri::URIDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/URI/URIDialectAttrDefs.cpp.inc"
      >();

  auto &blobInterface = addInterface<URIResourceManagerInterface>();
  addInterface<URIOpAsmInterface>(blobInterface);
}
