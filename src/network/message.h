// message.h
#ifndef MESSAGE_H
#define MESSAGE_H

#include <string>

class Message {
public:
    Message(const std::string& data)
        : data_(data) {}
    Message(const std::string& type, const std::string& data)
        : type_(type), data_(data) {}

    std::string getType() const { return type_; }
    std::string getData() const { return data_; }

    std::string toString() const { return type_ + ": " + data_; }

private:
    std::string type_;
    std::string data_;
};

#endif  // MESSAGE_H